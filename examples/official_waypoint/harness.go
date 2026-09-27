// External CPU fixture harness. Imports the complete, unchanged author package.
// Fixture files are not CRIU images; no process checkpoint or trainer is run.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	wp "github.com/Alex-XJK/waypoint/pkg/waypoint"
	"golang.org/x/sys/unix"
)

type fixture struct {
	manager *wp.Manager
	checkpoint, volatile, canonical, disk string
	files map[string][]byte
}

func must(err error) {
	if err != nil { panic(err) }
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func prepare(m *wp.Manager, base, ram, session, id string) fixture {
	f := fixture{manager:m, checkpoint:id,
		volatile:filepath.Join(ram, session, id),
		canonical:filepath.Join(base, session, "checkpoints", id, "criu"),
		disk:filepath.Join(base, session, "checkpoints", id, "criu.disk"),
		files:map[string][]byte{}}
	for index, name := range []string{"pages-fixture.img", "PAGES-fixture.img"} {
		data := make([]byte, 128 << 10)
		for i := range data { data[i] = byte((i*31+index*71)%251) }
		f.files[name] = data
	}
	f.files["manifest.json"] = []byte("{\"fixture_only\":true,\"criu_image\":false}\n")
	must(os.MkdirAll(f.volatile, 0755))
	must(os.MkdirAll(filepath.Dir(f.canonical), 0755))
	for name, data := range f.files { must(os.WriteFile(filepath.Join(f.volatile,name),data,0600)) }
	must(os.Symlink(f.volatile, f.canonical))
	return f
}

func verifyFiles(dir string, files map[string][]byte) error {
	entries, err := os.ReadDir(dir)
	if err != nil { return err }
	if len(entries) != len(files) { return fmt.Errorf("fixture file count differs") }
	for name, expected := range files {
		actual, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil { return err }
		if digest(actual) != digest(expected) { return fmt.Errorf("fixture bytes differ: %s", name) }
	}
	return nil
}

// Use genuine Linux flock on the same images.lock inode as the author API.
// This is a real reader lease, not a replacement for any author implementation.
func reader(f fixture) (string, error) {
	lock, err := os.OpenFile(filepath.Join(filepath.Dir(f.canonical),"images.lock"),os.O_CREATE|os.O_RDWR,0644)
	if err != nil { return "", err }
	defer lock.Close()
	if err := unix.Flock(int(lock.Fd()),unix.LOCK_SH); err != nil { return "",err }
	defer unix.Flock(int(lock.Fd()),unix.LOCK_UN)
	target, err := os.Readlink(f.canonical)
	if err != nil { return "",err }
	if target != f.volatile && target != "criu.disk" { return target,fmt.Errorf("noncanonical symlink target") }
	return target, verifyFiles(f.canonical,f.files)
}

func atomicHandoff(f fixture) map[string]any {
	started := time.Now()
	lock, err := os.OpenFile(filepath.Join(filepath.Dir(f.canonical),"images.lock"),os.O_CREATE|os.O_RDWR,0644)
	must(err)
	defer lock.Close()
	must(unix.Flock(int(lock.Fd()),unix.LOCK_SH))
	held := true
	defer func(){ if held { _ = unix.Flock(int(lock.Fd()),unix.LOCK_UN) } }()
	done := make(chan error,1)
	go func(){ done <- f.manager.FlushCheckpointImages(f.checkpoint) }()
	stop := make(chan struct{})
	var workers sync.WaitGroup
	var oldReads, newReads atomic.Int64
	var failuresMu sync.Mutex
	failures := []string{}
	readersClosed := false
	defer func(){
		if held { _ = unix.Flock(int(lock.Fd()),unix.LOCK_UN);held=false }
		if !readersClosed {close(stop);readersClosed=true}
		workers.Wait()
	}()
	for i:=0;i<2;i++ {
		workers.Add(1)
		go func(){
			defer workers.Done()
			for {
				select { case <-stop:return; default: }
				target, err := reader(f)
				if err != nil { failuresMu.Lock();failures=append(failures,err.Error());failuresMu.Unlock();return }
				if target == f.volatile { oldReads.Add(1) } else { newReads.Add(1) }
				time.Sleep(time.Millisecond)
			}
		}()
	}
	deadline := time.Now().Add(5*time.Second)
	for verifyFiles(f.disk,f.files)!=nil && time.Now().Before(deadline) { time.Sleep(time.Millisecond) }
	must(verifyFiles(f.disk,f.files))
	// Deliberately hold a real reader lease: this is a correctness barrier,
	// so the resulting wall time must not be advertised as a speed benchmark.
	select { case err:=<-done:panic(fmt.Sprintf("flush ended while shared lease held: %v",err));
	case <-time.After(40*time.Millisecond): }
	before, err := os.Readlink(f.canonical);must(err)
	if before != f.volatile { panic("canonical path switched before reader lease release") }
	must(verifyFiles(f.volatile,f.files))
	must(unix.Flock(int(lock.Fd()),unix.LOCK_UN));held=false
	select { case err:=<-done:must(err);case <-time.After(5*time.Second):panic("flush did not finish after reader release") }
	for i:=0;i<8;i++ { target,err:=reader(f);must(err);if target!="criu.disk" {panic("durable target not selected")};newReads.Add(1) }
	close(stop);readersClosed=true;workers.Wait()
	if len(failures)>0 || oldReads.Load()==0 { panic("concurrent exact-byte reader audit failed") }
	_, err = os.Stat(f.volatile)
	if !os.IsNotExist(err) { panic("volatile image directory retained after successful flush") }
	must(f.manager.FlushCheckpointImages(f.checkpoint))
	must(verifyFiles(f.canonical,f.files))
	return map[string]any{"case":"atomic_reader_handoff", "passed":true,
		"concurrent_readers":2,"old_target_exact_reads":oldReads.Load(),"new_target_exact_reads":newReads.Load(),
		"reader_errors":failures,"shared_lease_prevented_repoint":true,"source_removed_after_flush":true,
		"idempotent_second_flush":true,"injected_lease_hold_ms":40,
		"functional_case_wall_ms":float64(time.Since(started).Microseconds())/1000}
}

func failureRetry(f fixture) map[string]any {
	denied := filepath.Join(f.volatile,"pages-fixture.img")
	must(os.Chmod(denied,0000))
	err := f.manager.FlushCheckpointImages(f.checkpoint)
	if err==nil { panic("expected real EACCES from unreadable source fixture") }
	target, targetErr := os.Readlink(f.canonical);must(targetErr)
	if target!=f.volatile { panic("failed copy published a new canonical target") }
	partial, statErr := os.Stat(f.disk)
	partialLeft := statErr==nil && partial.IsDir()
	must(os.Chmod(denied,0600))
	must(verifyFiles(f.volatile,f.files))
	must(f.manager.FlushCheckpointImages(f.checkpoint))
	must(verifyFiles(f.canonical,f.files))
	target,targetErr=os.Readlink(f.canonical);must(targetErr)
	if target!="criu.disk" {panic("retry did not publish disk target")}
	return map[string]any{"case":"real_permission_failure_and_retry","passed":true,
		"failure_observed":err.Error(),"canonical_path_preserved_on_failure":true,
		"source_bytes_preserved":true,"partial_disk_directory_observed":partialLeft,
		"retry_replaces_partial_copy_and_preserves_all_bytes":true}
}

func main() {
	report:=map[string]any{"schema_version":"official-waypoint-imagestore-fixture-v1",
		"scope":"unchanged_author_images_store_with_owned_byte_fixtures",
		"actual_criu_run":false,"model_rollout_run":false,"trainer_run":false,
		"whole_process_checkpoint_verified":false,"speedup_claim":false,
		"go_version":runtime.Version(),"goos":runtime.GOOS,"goarch":runtime.GOARCH,"uid":os.Geteuid()}
	if build,ok:=debug.ReadBuildInfo();ok {report["build_info"]=build}
	failed:=false
	func(){
		defer func(){if value:=recover();value!=nil {failed=true;report["error"]=fmt.Sprint(value)}}()
		if runtime.GOOS!="linux" || os.Geteuid()==0 {panic("non_root_Linux_process_required")}
		// Keep the owned path short: the original manager enforces sun_path.
		root,err:=os.MkdirTemp("/tmp","wpd-");must(err)
		defer os.RemoveAll(root)
		ram,err:=os.MkdirTemp("/dev/shm","fpb-waypoint-ram-");must(err)
		defer os.RemoveAll(ram)
		var diskStat,ramStat unix.Statfs_t
		must(unix.Statfs(root,&diskStat));must(unix.Statfs(ram,&ramStat))
		if ramStat.Type!=unix.TMPFS_MAGIC || diskStat.Type==unix.TMPFS_MAGIC {panic("real_tmpfs_source_and_distinct_disk_filesystem_required")}
		report["fixture_filesystems"]=map[string]any{"source_tmpfs_magic":fmt.Sprintf("0x%x",ramStat.Type),
			"disk_filesystem_magic":fmt.Sprintf("0x%x",diskStat.Type),"source_is_tmpfs":true,"disk_is_tmpfs":false}
		base:=filepath.Join(root,"s")
		config:=filepath.Join(root,"empty-config.json");must(os.WriteFile(config,[]byte("{}\n"),0600))
		for name,value:=range map[string]string{"WAYPOINT_CONFIG":config,"WAYPOINT_SESSIONS_DIR":base,
			"WAYPOINT_SESSION_INFO_DIR":filepath.Join(root,"registry"),"WAYPOINT_TMPFS_IMAGES":"true","WAYPOINT_TMPFS_DIR":ram} {
			must(os.Setenv(name,value))
		}
		manager,session,err:=wp.NewManagerWithSession();must(err)
		first:=prepare(manager,base,ram,session,"atomic-fixture")
		hashes:=map[string]string{};fixtureBytes:=0
		for name,data:=range first.files {hashes[name]=digest(data);fixtureBytes+=len(data)}
		report["fixture_sha256"]=hashes;report["fixture_bytes_per_case"]=fixtureBytes
		report["cases"]=[]map[string]any{atomicHandoff(first),failureRetry(prepare(manager,base,ram,session,"retry-fixture"))}
		report["passed"]=true
	}()
	if failed {report["passed"]=false}
	encoder:=json.NewEncoder(os.Stdout);encoder.SetIndent("","  ");must(encoder.Encode(report))
	if failed {os.Exit(1)}
}
