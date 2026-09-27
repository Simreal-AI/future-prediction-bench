//! Owned external fixture; links the complete, unchanged cubecow crate normally.
//! No substitute snapshot implementation or full-copy fallback is provided.
use cubecow::{config::AppConfig, CubecowError, CubecowResult, Engine};
use std::collections::HashSet;
use serde_json::{json, Value};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MIB: u64 = 1024 * 1024;
const CHUNK: usize = 64 * 1024;
const SOURCE_PIN: &str = "d0081641c59822e4e5653b7462e914410b81910a";
const PHASES: [&str; 6] = ["populate", "mutate", "resize", "delete-origin", "recover-orphan", "cleanup"];

#[derive(Clone, Copy)]
enum Kind { Original, SourceMutated, BranchA, BranchB }

fn base_byte(offset: u64, size: u64) -> u8 {
    (((offset.wrapping_mul(73) ^ (offset >> 8) ^ (offset >> 20) ^ (size / MIB)) % 251) + 1) as u8
}

fn expected_byte(offset: u64, size: u64, kind: Kind, expanded: bool) -> u8 {
    if expanded && offset >= size { return 0; }
    let region = match kind {
        Kind::BranchA => Some((0, 31_u64)),
        Kind::BranchB => Some((size / 2, 67_u64)),
        Kind::SourceMutated => Some((size - CHUNK as u64, 113_u64)),
        Kind::Original => None,
    };
    if let Some((start, tag)) = region {
        if offset >= start && offset < start + CHUNK as u64 {
            return ((((offset - start).wrapping_mul(17) + tag) % 251) + 1) as u8;
        }
    }
    base_byte(offset, size)
}

fn expected_chunk(start: u64, n: usize, size: u64, kind: Kind, expanded: bool) -> Vec<u8> {
    (0..n).map(|i| expected_byte(start + i as u64, size, kind, expanded)).collect()
}

fn hash_output(stdout: &[u8]) -> Result<String> {
    let line = std::str::from_utf8(stdout)?;
    let digest = line.split_whitespace().next().ok_or("missing sha256sum output")?;
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid sha256sum digest".into());
    }
    Ok(digest.to_ascii_lowercase())
}

fn sha_file(path: &Path) -> Result<String> {
    let output = Command::new("/usr/bin/sha256sum").arg("--").arg(path).output()?;
    if !output.status.success() { return Err(format!("sha256sum file failed: {}", output.status).into()); }
    hash_output(&output.stdout)
}

fn sha_expected(size: u64, total: u64, kind: Kind, expanded: bool) -> Result<String> {
    let mut child = Command::new("/usr/bin/sha256sum").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    {
        let mut input = child.stdin.take().ok_or("missing sha256sum stdin")?;
        let mut offset = 0;
        while offset < total {
            let n = (total - offset).min(CHUNK as u64) as usize;
            input.write_all(&expected_chunk(offset, n, size, kind, expanded))?;
            offset += n as u64;
        }
    }
    let output = child.wait_with_output()?;
    if !output.status.success() { return Err("sha256sum expected stream failed".into()); }
    hash_output(&output.stdout)
}

fn verify(path: &Path, size: u64, kind: Kind, expanded: bool, role: &str) -> Result<Value> {
    let total = size + if expanded { MIB } else { 0 };
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.len() != total { return Err(format!("wrong file length/type for {role}").into()); }
    let mut checked = 0;
    let mut buffer = vec![0; CHUNK];
    while checked < total {
        let n = (total - checked).min(CHUNK as u64) as usize;
        file.read_exact(&mut buffer[..n])?;
        let expected = expected_chunk(checked, n, size, kind, expanded);
        if buffer[..n] != expected[..] {
            let first = buffer[..n].iter().zip(&expected).position(|(a,b)| a != b).unwrap();
            return Err(format!("byte mismatch for {role} at {}", checked + first as u64).into());
        }
        checked += n as u64;
    }
    if file.read(&mut buffer[..1])? != 0 { return Err("extra file byte".into()); }
    let actual = sha_file(path)?;
    let expected = sha_expected(size, total, kind, expanded)?;
    if actual != expected { return Err(format!("hash mismatch for {role}").into()); }
    Ok(json!({"role":role,"path":path,"logical_bytes":total,"allocated_bytes":meta.blocks()*512,"dev":meta.dev(),"inode":meta.ino(),"full_bytes_compared":checked,"sha256":actual,"expected_sha256":expected,"fsync_boundary":"writers used File::sync_all before verification; upstream FICLONE does not fsync destination"}))
}

fn engine(root: &Path) -> Result<Box<dyn Engine>> {
    let cfg = AppConfig::from_json_str(&json!({"log":{"format":"compact","rotation":"never"},"backend":{"kind":"reflink","reflink":{"root_dir":root}}}).to_string())?;
    Ok(cubecow::initialize_without_logging(cfg)?)
}

fn populate(path: &Path, size: u64) -> Result<()> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let mut offset = 0;
    while offset < size {
        let n = (size-offset).min(CHUNK as u64) as usize;
        file.write_all(&expected_chunk(offset,n,size,Kind::Original,false))?;
        offset += n as u64;
    }
    file.sync_all()?;
    Ok(())
}

fn mutate(path: &Path, size: u64, kind: Kind, start: u64) -> Result<()> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(start))?;
    file.write_all(&expected_chunk(start,CHUNK,size,kind,false))?;
    file.sync_all()?;
    Ok(())
}

fn name(prefix: &str, size: u64) -> String { format!("{prefix}-{}m", size/MIB) }

fn require_not_found<T>(result: CubecowResult<T>, label: &str) -> Result<()> {
    match result {
        Err(CubecowError::NotFound(_))=>Ok(()),
        Err(other)=>Err(format!("expected NotFound for {label}, received {other}").into()),
        Ok(_)=>Err(format!("deleted name still indexed: {label}").into()),
    }
}

fn all_witnesses(engine: &dyn Engine, size: u64, mutated: bool, resized: bool, origin_deleted: bool) -> Result<Vec<Value>> {
    let mut witnesses = Vec::new();
    let entries = [("source",if mutated {Kind::SourceMutated} else {Kind::Original}), ("snap",Kind::Original),("snap2",Kind::Original),("a",if mutated {Kind::BranchA} else {Kind::Original}),("b",if mutated {Kind::BranchB} else {Kind::Original})];
    for (prefix,kind) in entries {
        if origin_deleted && prefix=="source" { continue; }
        let info = engine.get_volume_info(&name(prefix,size))?;
        witnesses.push(verify(Path::new(&info.device_path),size,kind,resized&&prefix=="a",prefix)?);
    }
    Ok(witnesses)
}

fn phase(root: &Path, phase_name: &str) -> Result<Value> {
    let engine = engine(root)?; // Actual public constructor probes FICLONE and scans persisted namespace.
    let mut cases = Vec::new();
    for size in [8*MIB,64*MIB] {
        let source=name("source",size); let snap=name("snap",size); let snap2=name("snap2",size);
        let a=name("a",size); let b=name("b",size); let c=name("c",size);
        let mut case=json!({"size_bytes":size});
        match phase_name {
            "populate" => {
                let volume=engine.create_volume(&source,size)?;
                populate(Path::new(&volume.device_path),size)?;
                let s=engine.create_snapshot_from_volume(&source,&snap,false)?;
                let s2=engine.create_snapshot_from_volume(&snap,&snap2,false)?;
                if s.origin_volume!=source || s2.origin_volume!=source {return Err("unexpected snapshot origin flattening".into());}
                engine.create_volume_from_snapshot(&snap,&a)?;
                engine.create_volume_from_snapshot(&snap,&b)?;
                case["witnesses"]=json!(all_witnesses(engine.as_ref(),size,false,false,false)?);
                case["source_fully_nonzero"]=json!(true);
                case["snapshot_of_snapshot_origin"]=json!(s2.origin_volume);
                case["duplicate_name_rejected"]=json!(matches!(engine.create_volume(&a,size),Err(CubecowError::AlreadyExists(_))));
                if case["duplicate_name_rejected"]!=true {return Err("duplicate accepted".into());}
            }
            "mutate" => {
                case["before"]=json!(all_witnesses(engine.as_ref(),size,false,false,false)?);
                for (prefix,kind,offset) in [("a",Kind::BranchA,0),("b",Kind::BranchB,size/2),("source",Kind::SourceMutated,size-CHUNK as u64)] {
                    let info=engine.get_volume_info(&name(prefix,size))?;
                    mutate(Path::new(&info.device_path),size,kind,offset)?;
                }
                case["mutation_regions"]=json!({"a":[0,CHUNK],"b":[size/2,CHUNK as u64],"source":[size-CHUNK as u64,CHUNK as u64]});
                case["witnesses"]=json!(all_witnesses(engine.as_ref(),size,true,false,false)?);
            }
            "resize" => {
                case["before"]=json!(all_witnesses(engine.as_ref(),size,true,false,false)?);
                let result=engine.resize_volume(&a,size+MIB)?;
                if result!=(size,size+MIB) {return Err("wrong resize result".into());}
                // Match a caller's explicit data-persistence boundary after resize.
                File::open(&engine.get_volume_info(&a)?.device_path)?.sync_all()?;
                if !matches!(engine.resize_volume(&a,size),Err(CubecowError::InvalidArg(_))) {return Err("shrink must return InvalidArg".into());}
                case["resize_result"]=json!([result.0,result.1]);
                case["shrink_rejected"]=json!(true);
                case["witnesses"]=json!(all_witnesses(engine.as_ref(),size,true,true,false)?);
            }
            "delete-origin" => {
                case["before"]=json!(all_witnesses(engine.as_ref(),size,true,true,false)?);
                engine.delete_volume(&source)?;
                require_not_found(engine.get_volume_info(&source),&source)?;
                if !root.join("volumes").join(&source).is_dir() {return Err("snapshot origin directory removed early".into());}
                case["witnesses"]=json!(all_witnesses(engine.as_ref(),size,true,true,true)?);
                case["deleted_origin_not_found"]=json!(true);
                case["orphan_directory_preserved"]=json!(true);
            }
            "recover-orphan" => {
                require_not_found(engine.get_volume_info(&source),&source)?;
                case["before"]=json!(all_witnesses(engine.as_ref(),size,true,true,true)?);
                let (snapshots,token)=engine.list_snapshots(&source,0,None);
                // Upstream intentionally lists an empty page for a deleted origin.
                // Recovery is instead available through each canonical snapshot name.
                if !snapshots.is_empty() || token.is_some() {return Err("deleted origin snapshot page should be empty".into());}
                let recovered_names=vec![engine.get_volume_info(&snap)?.name,engine.get_volume_info(&snap2)?.name];
                if recovered_names!=vec![snap.clone(),snap2.clone()] {return Err("canonical snapshot recovery mismatch".into());}
                let recovered=engine.create_volume_from_snapshot(&snap,&c)?;
                case["deleted_origin_list_is_empty"]=json!(true);
                case["canonical_recovered_snapshot_names"]=json!(recovered_names);
                case["new_branch_from_orphan"]=verify(Path::new(&recovered.device_path),size,Kind::Original,false,"c")?;
                engine.delete_snapshot(&snap)?;
                case["remaining_snapshot"]=verify(Path::new(&engine.get_volume_info(&snap2)?.device_path),size,Kind::Original,false,"snap2")?;
                engine.delete_snapshot(&snap2)?;
                if root.join("volumes").join(&source).exists() {return Err("last snapshot deletion left orphan directory".into());}
                for removed in [&source,&snap,&snap2] {require_not_found(engine.get_volume_info(removed),removed)?;}
                case["last_snapshot_reaped_origin_directory"]=json!(true);
                case["survivors"]=json!([
                    verify(Path::new(&engine.get_volume_info(&a)?.device_path),size,Kind::BranchA,true,"a")?,
                    verify(Path::new(&engine.get_volume_info(&b)?.device_path),size,Kind::BranchB,false,"b")?,
                    verify(Path::new(&engine.get_volume_info(&c)?.device_path),size,Kind::Original,false,"c")?
                ]);
            }
            "cleanup" => {
                for removed in [&source,&snap,&snap2] {require_not_found(engine.get_volume_info(removed),removed)?;}
                case["before"]=json!([
                    verify(Path::new(&engine.get_volume_info(&a)?.device_path),size,Kind::BranchA,true,"a")?,
                    verify(Path::new(&engine.get_volume_info(&b)?.device_path),size,Kind::BranchB,false,"b")?,
                    verify(Path::new(&engine.get_volume_info(&c)?.device_path),size,Kind::Original,false,"c")?
                ]);
                for volume in [&a,&b,&c] {engine.delete_volume(volume)?;require_not_found(engine.get_volume_info(volume),volume)?;}
                case["deleted_all_branches"]=json!(true);
            }
            _=>return Err("unknown phase".into()),
        }
        cases.push(case);
    }
    if phase_name=="cleanup" {
        let (volumes,token,total)=engine.list_volumes(0,None);
        if !volumes.is_empty() || token.is_some() || total!=0 || fs::read_dir(root.join("volumes"))?.next().is_some() {return Err("cleanup namespace not empty".into());}
    }
    Ok(json!({"phase":phase_name,"pid":std::process::id(),"official_source_commit":SOURCE_PIN,"cases":cases,"actual_public_constructor":true,"silently_skipped":false}))
}

fn entry() -> Result<()> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()==4 && args[1]=="--phase" {
        println!("{}",phase(Path::new(&args[3]),&args[2])?);
        return Ok(());
    }
    if args.len()!=5 || args[1]!="--root" || args[3]!="--report" {return Err("usage: fpb-cubecow-populated-fixture --root ABSOLUTE_FRESH_XFS_ROOT --report ABSOLUTE_FRESH_JSON".into());}
    let root=PathBuf::from(&args[2]);let report=PathBuf::from(&args[4]);
    if !root.is_absolute() || !report.is_absolute() || root.exists() || report.exists() || root==report || report.starts_with(&root) {return Err("require disjoint absolute fresh root/report paths".into());}
    fs::create_dir(&root)?; // The parent must be the owned, actually mounted XFS filesystem.
    let executable=std::env::current_exe()?;
    let mut results=Vec::new();
    let mut failure=None;
    let mut process_ids=HashSet::new();
    for phase_name in PHASES {
        let child=Command::new(&executable).args(["--phase",phase_name]).arg(&root).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
        let spawned_pid=child.id();
        let output=child.wait_with_output()?;
        let parsed=serde_json::from_slice::<Value>(&output.stdout);
        let record=json!({"phase":phase_name,"spawned_pid":spawned_pid,"returncode":output.status.code(),"stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr),"result":parsed.ok()});
        if !output.status.success() || record["result"]["pid"]!=spawned_pid || record["result"]["phase"]!=phase_name || !process_ids.insert(spawned_pid) || spawned_pid==std::process::id(){failure=Some(phase_name.to_string());}
        results.push(record);
        if failure.is_some(){break;}
    }
    let result=json!({"status":if failure.is_none(){"pass"}else{"fail"},"failed_phase":failure,"official_source_commit":SOURCE_PIN,"fixture_executable":executable,"fixture_executable_sha256":sha_file(&executable)?,"sha256sum_executable":"/usr/bin/sha256sum","sha256sum_executable_sha256":sha_file(Path::new("/usr/bin/sha256sum"))?,"master_pid":std::process::id(),"separate_native_process_phases":results,"runtime_scope":"actual unchanged cubecow public library filesystem reflink; no VM/process-memory checkpoint, model rollout, optimizer or training throughput result","timing_or_speedup_claimed":false});
    let mut output=OpenOptions::new().write(true).create_new(true).open(&report)?;
    output.write_all(serde_json::to_string_pretty(&result)?.as_bytes())?;output.write_all(b"\n")?;output.sync_all()?;
    println!("{}",json!({"status":result["status"],"report":report,"completed_phases":results.len()}));
    if result["status"]!="pass" {return Err("a native phase failed; raw output preserved".into());}
    Ok(())
}

fn main() {
    if let Err(error)=entry(){eprintln!("fixture error: {error}");std::process::exit(1);}
}
