//! Owned component benchmark. All snapshot/fork operations call the unchanged
//! original library. The baseline is explicit userspace read/write full copy.
use cubecow::{config::AppConfig, Engine};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MIB:u64=1048576;
const BLOCK:usize=65536;
const WARMUP_PAIRS:usize=2;
const MEASURED_PAIRS:usize=8;
const PIN:&str="d0081641c59822e4e5653b7462e914410b81910a";
const INDEPENDENT_ORACLE_EVIDENCE:&str="82ec9a8dbb1f7055f282b029d0bafd35b4e75c7b9420d985f7c6f39a47420a9a";

fn digest_expected(size:u64,kind:&str)->Result<&'static str>{
    match(size/MIB,kind){
        (8,"base")=>Ok("d203aa98a4525db8c356a38098e44cccc998195281ba810466f23cb3577f1a7e"),
        (64,"base")=>Ok("f14951d41351d8f42d1e91a579226d288ee8fb740b28846e87b5eee234786c92"),
        (8,"prefix")=>Ok("e9496e71926535b3e2230d51b6e8788c46ec9bc8ead6b49810eba1092fe7ccc2"),
        (64,"prefix")=>Ok("b9f644ef4b30d46e652e341bd93e36422c4e44a9240c8b81ae4422eab71fef18"),
        (8,"middle")=>Ok("f9df3f081e35b420ba290978be3b02d51b0b0c73bdbaf9a885492ee7cdabacf1"),
        (64,"middle")=>Ok("6349192744d3e85ad6580d48f1ececc642f7d5a21f2baa8425bf5f7e53c83c6e"),
        (8,"suffix")=>Ok("cb488e44efee70943569c6d6e8145e04be930b4bb481d14858545bfea4508b13"),
        (64,"suffix")=>Ok("5214d00813aac9cbf00d8956fbab12398f1a27ab07cb5793546f93f0f7e59e9b"),
        _=>Err("unsupported independently prepared digest".into())
    }
}

fn sha(path:&Path)->Result<String>{
    let output=Command::new("/usr/bin/sha256sum").arg("--").arg(path).output()?;
    if !output.status.success(){return Err("real sha256sum failed".into());}
    let text=std::str::from_utf8(&output.stdout)?;
    let digest=text.split_whitespace().next().ok_or("missing SHA output")?;
    if digest.len()!=64 || !digest.bytes().all(|b|b.is_ascii_hexdigit()){return Err("bad SHA output".into());}
    Ok(digest.to_ascii_lowercase())
}

fn verify(path:&Path,size:u64,kind:&str,role:&str)->Result<Value>{
    let meta=fs::metadata(path)?;
    if !meta.is_file() || meta.len()!=size{return Err(format!("wrong data length for {role}").into());}
    let actual=sha(path)?;let expected=digest_expected(size,kind)?;
    if actual!=expected{return Err(format!("whole-file data mismatch for {role}").into());}
    Ok(json!({"role":role,"path":path,"size_bytes":size,"kind":kind,"sha256":actual,"independent_expected_sha256":expected,"dev":meta.dev(),"inode":meta.ino(),"allocated_bytes":meta.blocks()*512}))
}

fn base(offset:u64,size:u64)->u8{(((offset*73 ^ (offset>>8) ^ (offset>>20) ^ (size/MIB))%251)+1) as u8}

fn populate(path:&Path,size:u64)->Result<()>{
    let mut file=OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut offset=0;
    while offset<size{
        let n=(size-offset).min(BLOCK as u64) as usize;
        let data=(0..n).map(|i|base(offset+i as u64,size)).collect::<Vec<_>>();
        file.write_all(&data)?;offset+=n as u64;
    }
    file.sync_all()?;Ok(())
}

fn write_patch(path:&Path,size:u64,offset:u64,tag:Option<u64>)->Result<()>{
    let mut file=OpenOptions::new().write(true).open(path)?;file.seek(SeekFrom::Start(offset))?;
    let data=(0..BLOCK).map(|i|match tag{Some(t)=>(((i as u64*17+t)%251)+1) as u8,None=>base(offset+i as u64,size)}).collect::<Vec<_>>();
    file.write_all(&data)?;file.sync_all()?;Ok(())
}

fn sync_dir(path:&Path)->Result<()>{File::open(path)?.sync_all()?;Ok(())}

fn explicit_copy(source:&Path,destination:&Path,size:u64)->Result<()>{
    let mut input=File::open(source)?;
    if input.metadata()?.len()!=size{return Err("full-copy source size mismatch".into());}
    let mut output=OpenOptions::new().write(true).create_new(true).open(destination)?;
    let mut data=vec![0_u8;BLOCK];let mut copied=0;
    loop{let n=input.read(&mut data)?;if n==0{break;}output.write_all(&data[..n])?;copied+=n as u64;}
    if copied!=size || output.metadata()?.len()!=size{return Err("full-copy materialization incomplete".into());}
    // Data fsync is intentionally absent at the API-return boundary, matching
    // upstream FICLONE. Directory-entry fsyncs mirror the upstream API below.
    Ok(())
}

struct Prepared{
    size:u64,source_name:String,reference_name:String,
    official_source:PathBuf,official_reference:PathBuf,
    copy_source:PathBuf,copy_reference:PathBuf,
}

fn prepare(engine:&dyn Engine,official:&Path,copy:&Path,size:u64)->Result<Prepared>{
    let source_name=format!("source-{}m",size/MIB);let reference_name=format!("reference-{}m",size/MIB);
    let created=engine.create_volume(&source_name,size)?;let official_source=PathBuf::from(created.device_path);
    // Original API creates the sparse main file. Populate it explicitly rather
    // than replacing the original create_volume implementation.
    let mut f=OpenOptions::new().write(true).open(&official_source)?;let mut at=0;
    while at<size{let n=(size-at).min(BLOCK as u64) as usize;let data=(0..n).map(|i|base(at+i as u64,size)).collect::<Vec<_>>();f.write_all(&data)?;at+=n as u64;}f.sync_all()?;
    let reference=engine.create_snapshot_from_volume(&source_name,&reference_name,false)?;
    let official_reference=PathBuf::from(reference.device_path);
    let dir=copy.join("volumes").join(&source_name);fs::create_dir_all(&dir)?;
    let copy_source=dir.join(&source_name);populate(&copy_source,size)?;
    let copy_reference=dir.join(&reference_name);explicit_copy(&copy_source,&copy_reference,size)?;File::open(&copy_reference)?.sync_all()?;
    for path in [&official_source,&official_reference,&copy_source,&copy_reference]{verify(path,size,"base","prepared-source")?;File::open(path)?.sync_all()?;}
    let identities=[&official_source,&official_reference,&copy_source,&copy_reference].iter().map(|path|{let meta=fs::metadata(path)?;Ok((meta.dev(),meta.ino()))}).collect::<std::io::Result<BTreeSet<_>>>()?;
    if identities.len()!=4{return Err("prepared original/reference/copy files alias inodes".into());}
    for path in [official.join("volumes").join(&source_name),official.join("volumes"),official.to_path_buf(),dir,copy.join("volumes"),copy.to_path_buf()]{sync_dir(&path)?;}
    Ok(Prepared{size,source_name,reference_name,official_source,official_reference,copy_source,copy_reference})
}

struct Created{files:Vec<(String,PathBuf)>,snapshot_name:Option<String>,fork_name:Option<String>}

fn execute(engine:&dyn Engine,official:&Path,copy:&Path,p:&Prepared,method:&str,operation:&str,label:&str)->Result<Created>{
    let snap_name=format!("snap-{label}");let fork_name=format!("fork-{label}");
    let mut made=Created{files:vec![],snapshot_name:None,fork_name:None};
    if method=="original"{
        let source=if operation=="fork"{&p.reference_name}else{&p.source_name};
        if operation=="snapshot" || operation=="checkpoint+fork"{
            let snapshot=engine.create_snapshot_from_volume(source,&snap_name,false)?;
            made.files.push(("snapshot".into(),PathBuf::from(snapshot.device_path)));made.snapshot_name=Some(snap_name.clone());
        }
        if operation=="fork" || operation=="checkpoint+fork"{
            let source=if operation=="fork"{&p.reference_name}else{&snap_name};
            let fork=engine.create_volume_from_snapshot(source,&fork_name)?;
            made.files.push(("branch".into(),PathBuf::from(fork.device_path)));made.fork_name=Some(fork_name);
        }
    }else{
        let source=if operation=="fork"{&p.copy_reference}else{&p.copy_source};
        let mut fork_source=source.to_path_buf();
        if operation=="snapshot" || operation=="checkpoint+fork"{
            let path=copy.join("volumes").join(&p.source_name).join(&snap_name);
            explicit_copy(source,&path,p.size)?;sync_dir(path.parent().unwrap())?;
            fork_source=path.clone();made.files.push(("snapshot".into(),path));made.snapshot_name=Some(snap_name);
        }
        if operation=="fork" || operation=="checkpoint+fork"{
            let dir=copy.join("volumes").join(&fork_name);fs::create_dir_all(&dir)?;let path=dir.join(&fork_name);
            explicit_copy(&fork_source,&path,p.size)?;sync_dir(&dir)?;sync_dir(&copy.join("volumes"))?;
            made.files.push(("branch".into(),path));made.fork_name=Some(fork_name);
        }
    }
    let _=official; // Each original returned path is validated below.
    Ok(made)
}

struct PersistenceScope{files:Vec<PathBuf>,dirs:Vec<PathBuf>}

fn caller_durable_scope(created:&Created,namespace:&Path,root:&Path)->Result<PersistenceScope>{
    let mut dirs=BTreeSet::new();let mut files=Vec::new();
    for(_,path)in &created.files{
        if !path.starts_with(namespace.join("volumes")){return Err("destination escaped owned namespace".into());}
        File::open(path)?.sync_all()?;files.push(path.clone());
        let mut parent=path.parent().ok_or("destination has no parent")?;
        loop{dirs.insert(parent.to_path_buf());if parent==root.parent().ok_or("fixture root has no parent")?{break;}parent=parent.parent().ok_or("ancestor persistence scope escaped root")?;}
    }
    let mut ordered=dirs.into_iter().collect::<Vec<_>>();ordered.sort_by_key(|p|std::cmp::Reverse(p.components().count()));
    for dir in &ordered{sync_dir(dir)?;}
    Ok(PersistenceScope{files,dirs:ordered})
}

fn cleanup(engine:&dyn Engine,created:&Created,method:&str)->Result<()>{
    if method=="original"{
        if let Some(name)=&created.fork_name{engine.delete_volume(name)?;}
        if let Some(name)=&created.snapshot_name{engine.delete_snapshot(name)?;}
    }else{
        for(role,path)in created.files.iter().rev(){fs::remove_file(path)?;if role=="branch"{fs::remove_dir(path.parent().unwrap())?;}}
    }
    Ok(())
}

fn run_trial(engine:&dyn Engine,official:&Path,copy:&Path,root:&Path,p:&Prepared,method:&str,operation:&str,mode:&str,label:&str,record:&mut Value)->Result<()>{
    let protected_source=if method=="original"{&p.official_source}else{&p.copy_source};
    let protected_reference=if method=="original"{&p.official_reference}else{&p.copy_reference};
    // Validation warms both source/reference data before each operation. This
    // is deliberately a warm-cache component experiment, not cold I/O.
    record["before"]=json!([verify(protected_source,p.size,"base","source")?,verify(protected_reference,p.size,"base","prepared-reference")?]);
    let metrics_before=engine.metrics();
    let total=*metrics_before.get("total_bytes").ok_or("original filesystem total metric absent")?;
    let used=*metrics_before.get("used_bytes").ok_or("original filesystem used metric absent")?;
    let worst_new=p.size*if operation=="checkpoint+fork"{2}else{1};
    if total==0 || total>512*MIB || used+worst_new+8*MIB>total{return Err("owned512MiB filesystem free-space guard rejected trial".into());}
    record["filesystem_space_before"]=json!({"total_bytes":total,"used_bytes":used,"worst_new_destination_bytes":worst_new,"reserved_headroom_bytes":8*MIB,"source":"actual original Engine::metrics/statvfs outside timing"});
    let started=Instant::now();
    let attempt=execute(engine,official,copy,p,method,operation,label);
    let returned=started.elapsed().as_nanos() as u64;
    // Stop the return timer before allocating or serializing any report data.
    let created=match attempt{Ok(created)=>created,Err(error)=>{record["operation_return_ns"]=json!(returned);return Err(error);}};
    let mut persistence=None;
    let mut durable_total=None;
    if mode=="caller-durable"{
        let namespace=if method=="original"{official}else{copy};
        let scope_result=caller_durable_scope(&created,namespace,root);
        let durable=started.elapsed().as_nanos() as u64;
        let scope=match scope_result{Ok(scope)=>scope,Err(error)=>{record["operation_return_ns"]=json!(returned);record["failed_caller_durable_elapsed_ns"]=json!(durable);record["cleanup_completed"]=json!(cleanup(engine,&created,method).is_ok());return Err(error);}};
        durable_total=Some(durable);persistence=Some(scope);
    }
    record["operation_return_ns"]=json!(returned);
    if let Some(total)=durable_total{record["caller_durable_total_ns"]=json!(total);record["caller_durable_extra_ns"]=json!(total-returned);}
    if let Some(scope)=&persistence{record["caller_durable_scope"]=json!({"data_files_synced":scope.files,"ancestor_directories_synced":scope.dirs,"scope_stop":root.parent(),"extra_separate_index_metadata_files":0,"source":"original self-describing files+directory namespace; no whole-filesystem sync or cache drop"});}
    let validation=(||->Result<()>{
        let namespace=if method=="original"{official}else{copy};
        for(role,path)in &created.files{
            let expected=if role=="snapshot"{namespace.join("volumes").join(&p.source_name).join(created.snapshot_name.as_ref().ok_or("snapshot name absent")?)}else{let name=created.fork_name.as_ref().ok_or("fork name absent")?;namespace.join("volumes").join(name).join(name)};
            if *path!=expected{return Err("original/baseline destination layout mismatch".into());}
        }
        // Original reflink namespace has no persisted separate index. Inspect
        // the actual generated paths outside timers and reject unknown files.
        let mut expected_files=BTreeSet::from([protected_source.to_path_buf(),protected_reference.to_path_buf()]);
        for(_,path)in &created.files{expected_files.insert(path.clone());}
        let mut expected_dirs=BTreeSet::new();let mut observed_files=BTreeSet::new();let mut observed_dirs=BTreeSet::new();
        for dir in fs::read_dir(namespace.join("volumes"))?{let dir=dir?;if !dir.file_type()?.is_dir(){return Err("unexpected namespace metadata entry".into());}observed_dirs.insert(dir.path());for file in fs::read_dir(dir.path())?{let file=file?;if !file.file_type()?.is_file(){return Err("unexpected generated metadata directory".into());}observed_files.insert(file.path());}}
        // Sources/references for both fixed sizes are long-lived and valid.
        for size in[8*MIB,64*MIB]{let origin=format!("source-{}m",size/MIB);let dir=namespace.join("volumes").join(&origin);expected_dirs.insert(dir.clone());expected_files.insert(dir.join(&origin));expected_files.insert(dir.join(format!("reference-{}m",size/MIB)));}
        if let Some(name)=&created.fork_name{expected_dirs.insert(namespace.join("volumes").join(name));}
        if expected_files!=observed_files || expected_dirs!=observed_dirs{return Err("unexpected metadata file/directory omitted from common persistence contract".into());}
        record["generated_layout_has_no_unaccounted_metadata_files"]=json!(true);
        let mut witnesses=Vec::new();
        for(role,path)in &created.files{witnesses.push(verify(path,p.size,"base",role)?);}
        record["at_return_validation"]=json!(witnesses);
        let final_path=&created.files.last().ok_or("operation produced no data")?.1;
        write_patch(final_path,p.size,0,Some(31))?;
        let mut isolated=vec![verify(final_path,p.size,"prefix","mutated-destination")?,verify(protected_source,p.size,"base","source-after-destination-write")?,verify(protected_reference,p.size,"base","reference-after-destination-write")?];
        if operation=="checkpoint+fork"{
            let snapshot=&created.files[0].1;
            isolated.push(verify(snapshot,p.size,"base","snapshot-after-branch-write")?);
            write_patch(snapshot,p.size,p.size/2,Some(67))?;
            isolated.push(verify(snapshot,p.size,"middle","mutated-snapshot")?);
            isolated.push(verify(final_path,p.size,"prefix","branch-after-snapshot-write")?);
        }
        // Exercise the other direction too, then restore the protected source
        // outside timing. A prepared fork source is the reference snapshot.
        let upstream=if operation=="fork"{protected_reference}else{protected_source};
        write_patch(upstream,p.size,p.size-BLOCK as u64,Some(113))?;
        let reverse=(||->Result<Value>{Ok(json!([verify(upstream,p.size,"suffix","mutated-upstream")?,verify(final_path,p.size,"prefix","destination-after-upstream-write")?]))})();
        write_patch(upstream,p.size,p.size-BLOCK as u64,None)?;
        record["upstream_write_isolation"]=reverse?;
        isolated.push(verify(protected_source,p.size,"base","restored-source")?);isolated.push(verify(protected_reference,p.size,"base","restored-reference")?);
        record["destination_write_isolation"]=json!(isolated);
        Ok(())
    })();
    let cleaned=cleanup(engine,&created,method);
    record["cleanup_completed"]=json!(cleaned.is_ok());
    record["filesystem_space_after_cleanup"]=json!(engine.metrics());
    validation?;cleaned?;
    record["status"]=json!("pass");Ok(())
}

fn distribution(values:&[u64])->Value{
    let mut sorted=values.to_vec();sorted.sort_unstable();let n=sorted.len();
    if n==0{return Value::Null;}
    let median=(sorted[(n-1)/2] as f64+sorted[n/2] as f64)/2.0;
    let p95=sorted[((n*95+99)/100).saturating_sub(1).min(n-1)];
    json!({"count":n,"minimum_ns":sorted[0],"median_ns":median,"mean_ns":values.iter().map(|v|*v as f64).sum::<f64>()/n as f64,"p95_ns":p95,"maximum_ns":sorted[n-1],"raw_ns":values})
}

fn persist(report:&Value,file:&mut File)->Result<()>{file.seek(SeekFrom::Start(0))?;file.set_len(0)?;file.write_all(serde_json::to_string_pretty(report)?.as_bytes())?;file.write_all(b"\n")?;file.sync_all()?;Ok(())}

fn main_result()->Result<()>{
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=5 || args[1]!="--root" || args[3]!="--report"{return Err("usage: fpb-cubecow-latency-fixture --root ABSOLUTE_FRESH_XFS_ROOT --report ABSOLUTE_FRESH_REPORT_OUTSIDE_ROOT".into());}
    let root=PathBuf::from(&args[2]);let report_path=PathBuf::from(&args[4]);
    if !root.is_absolute() || !report_path.is_absolute() || root.exists() || report_path.exists() || report_path.starts_with(&root){return Err("require disjoint fresh absolute owned paths".into());}
    fs::create_dir(&root)?;let official=root.join("official");let copy=root.join("copy");fs::create_dir_all(copy.join("volumes"))?;
    let cfg=AppConfig::from_json_str(&json!({"log":{"format":"compact","rotation":"never"},"backend":{"kind":"reflink","reflink":{"root_dir":official}}}).to_string())?;
    let engine=cubecow::initialize_without_logging(cfg)?;
    let mut prepared=Vec::new();for size in[8*MIB,64*MIB]{prepared.push(prepare(engine.as_ref(),&official,&copy,size)?);}
    sync_dir(&root)?;sync_dir(root.parent().ok_or("no root parent")?)?;
    let executable=std::env::current_exe()?;
    let mut report=json!({"status":"running","source_commit":PIN,"fixture_executable":executable,"fixture_executable_sha256":sha(&executable)?,"coreutils_executable_sha256":sha(Path::new("/usr/bin/sha256sum"))?,"independent_pattern_digest_evidence_sha256":INDEPENDENT_ORACLE_EVIDENCE,"warmup_pairs":WARMUP_PAIRS,"measured_pairs":MEASURED_PAIRS,"sizes_bytes":[8*MIB,64*MIB],"cache_contract":"warm-cache, full source/reference validation before each trial; no cache dropping","baseline":"explicit64KiB userspace File::read/write_all full materialization; no std::fs::copy or copy_file_range","return_contract":"original API includes namespace/indexing and best-effort directory fsync; baseline mirrors directory-entry fsync but has no engine indexing","durable_contract":"operation then caller sync_all of every new data file and every ancestor directory through the owned XFS mount root; no separate persisted index in original layout","scope":"filesystem snapshot/fork components only; no originalCubeVMM/process-memory/model/token/optimizer/GPU throughput claim","planned_max_live_payload_bytes":344*MIB,"samples":[],"groups":[]});
    let mut output=OpenOptions::new().write(true).create_new(true).open(&report_path)?;persist(&report,&mut output)?;
    for p in &prepared{for mode in["return","caller-durable"]{for operation in["snapshot","fork","checkpoint+fork"]{
        let mut group_samples=Vec::new();
        for pair in 0..WARMUP_PAIRS+MEASURED_PAIRS{
            let warmup=pair<WARMUP_PAIRS;
            let order=if pair%2==0{["original","full-copy"]}else{["full-copy","original"]};
            for(slot,method)in order.iter().enumerate(){
                let label=format!("{}m-{}-{}-p{}-{}",p.size/MIB,mode,operation.replace('+',"_"),pair,method);
                let mut sample=json!({"status":"running","size_bytes":p.size,"mode":mode,"operation":operation,"pair":pair,"warmup":warmup,"order":order,"order_slot":slot,"method":method,"operation_return_ns":null,"caller_durable_total_ns":null,"caller_durable_extra_ns":null});
                if let Err(error)=run_trial(engine.as_ref(),&official,&copy,&root,p,method,operation,mode,&label,&mut sample){sample["status"]=json!("fail");sample["error"]=json!(error.to_string());}
                group_samples.push(sample.clone());report["samples"].as_array_mut().unwrap().push(sample.clone());
                if sample["status"]!="pass"{report["status"]=json!("fail");persist(&report,&mut output)?;return Err("trial failed; all performed samples preserved, no ratio accepted".into());}
            }
            // Report persistence and cleanup are outside all primitive timers.
            persist(&report,&mut output)?;
        }
        let mut statistics=serde_json::Map::new();
        for method in["original","full-copy"]{
            let measured=group_samples.iter().filter(|s|s["warmup"]==false && s["method"]==method).collect::<Vec<_>>();
            let returned=measured.iter().map(|s|s["operation_return_ns"].as_u64().unwrap()).collect::<Vec<_>>();
            let durable=measured.iter().filter_map(|s|s["caller_durable_total_ns"].as_u64()).collect::<Vec<_>>();
            statistics.insert(method.into(),json!({"operation_return":distribution(&returned),"caller_durable_total":distribution(&durable)}));
        }
        report["groups"].as_array_mut().unwrap().push(json!({"size_bytes":p.size,"mode":mode,"operation":operation,"all_pairs_accepted":true,"warmup_pairs":WARMUP_PAIRS,"measured_pairs":MEASURED_PAIRS,"original_first_measured_pairs":4,"copy_first_measured_pairs":4,"statistics":statistics}));
    }}}
    if report["samples"].as_array().unwrap().len()!=240 || report["groups"].as_array().unwrap().len()!=12{return Err("fixed matrix incomplete".into());}
    for p in &prepared{engine.delete_snapshot(&p.reference_name)?;engine.delete_volume(&p.source_name)?;fs::remove_file(&p.copy_reference)?;fs::remove_file(&p.copy_source)?;fs::remove_dir(p.copy_source.parent().unwrap())?;}
    if fs::read_dir(official.join("volumes"))?.next().is_some() || fs::read_dir(copy.join("volumes"))?.next().is_some(){return Err("managed namespace cleanup incomplete".into());}
    report["status"]=json!("pass");report["cleanup_all_prepared_data_completed"]=json!(true);persist(&report,&mut output)?;
    println!("{}",json!({"status":"pass","report":report_path,"groups":12,"all_method_trials":240,"measured_method_trials":192,"scope":"actual filesystem component timings; no end-to-end training claim"}));Ok(())
}

fn main(){if let Err(error)=main_result(){eprintln!("latency fixture error: {error}");std::process::exit(1);}}
