"""Unprivileged Linux Landlock restriction for a trusted frozen selector.

No host configuration changes or privilege escalation. Exact pre-existing output
inodes avoid a readable output-directory grant. Python audit is supplemental;
Landlock mediates native opens too. This is not a full hostile-code sandbox.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import platform
import stat
import sys

CREATE,ADD,RESTRICT=444,445,446 # Linux x86_64 syscall ABI.
READ_FILE=1<<2;READ_DIR=1<<3;WRITE_FILE=1<<1;TRUNCATE=1<<14
REFER=1<<13;NO_NEW_PRIVS=38;TSYNC=1<<3

class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64),('handled_access_net',ctypes.c_uint64),('scoped',ctypes.c_uint64)]
class Beneath(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]


def libc():
    if platform.system()!='Linux' or platform.machine()!='x86_64': raise RuntimeError('Only reviewed Linux x86_64 ABI is supported')
    lib=ctypes.CDLL(None,use_errno=True);lib.syscall.restype=ctypes.c_long
    lib.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong];lib.prctl.restype=ctypes.c_int
    return lib

def abi_version():
    lib=libc();value=lib.syscall(CREATE,None,0,1)
    if value<0: raise OSError(ctypes.get_errno(),'Landlock ABI query failed')
    if value<6: raise RuntimeError('Landlock ABI >=6 required; no degraded selector execution')
    return int(value)

def _identity(path):
    s=Path(path).stat();return s.st_dev,s.st_ino

def _regular(path):
    p=Path(path)
    if p.is_symlink(): raise ValueError('symlink is not an admissible exact file: '+str(p))
    s=p.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1: raise ValueError('exact file must be regular with one link: '+str(p))
    return p.resolve()

def validate_run(run,expected_names,protected_paths):
    run=Path(run)
    if run.is_symlink() or not run.is_dir(): raise ValueError('invalid selector output directory')
    if set(p.name for p in run.iterdir())!=set(expected_names): raise ValueError('unexpected selector run contents')
    identities={_identity(p) for p in protected_paths if Path(p).exists()}
    for p in run.iterdir():
        resolved=_regular(p)
        if _identity(resolved) in identities: raise ValueError('raw-source inode alias in selector run')
    return True

def build_policy(project,run,read_files,write_files,protected_paths):
    project=Path(project).resolve();run=Path(run).resolve();src=project/'src'
    runtime=[Path('/usr/lib').resolve()]
    prefix=Path(sys.prefix).resolve()
    if prefix not in (Path('/usr'),Path('/')): runtime.append(prefix)
    dirs=sorted(set(str(p) for p in [*runtime,src] if p.is_dir()))
    raw=[str(Path(p).resolve()) for p in protected_paths]
    forbidden={_identity(p) for p in raw if Path(p).exists()}
    for p in raw:
        if any(Path(p).is_relative_to(Path(d)) for d in dirs): raise ValueError('raw source under a readable runtime/source directory')
    # The project source tree is small and mutable; disallow hidden data aliases.
    if src.exists():
        for p in src.rglob('*'):
            if p.is_symlink(): raise ValueError('symlink in project source tree')
            if p.is_file() and (p.stat().st_nlink!=1 or _identity(p) in forbidden): raise ValueError('hardlink/raw alias in project source tree')
    reads=[];writes=[]
    for collection,names in ((reads,read_files),(writes,write_files)):
        for name in names:
            path=_regular(name)
            if not path.is_relative_to(project) or path.is_relative_to(project/'data') or path.is_relative_to(Path('/proc')):
                raise ValueError('selector exact file outside approved workspace artifacts')
            if _identity(path) in forbidden: raise ValueError('protected inode cannot be allowed')
            collection.append(str(path))
    for name in writes:
        path=Path(name)
        if path.parent!=run or path.stat().st_size!=0: raise ValueError('outputs must be empty preallocated run files')
    return {'version':1,'read_directories':dirs,'read_files':sorted(set(reads)),'write_files':sorted(set(writes)),
            'protected_paths':raw,'run_directory_read_granted':False,'directory_write_create_link_granted':False,
            'trusted_runtime_directories':[str(p) for p in runtime],
            'limitations':['Runtime directories and frozen source code are trusted; external mutation/path races are outside this bounded selector threat model.',
             'Landlock does not restrict all metadata operations or previously opened descriptors; no /proc read grant is made.',
             'ABI 8 restricts TCP and abstract UNIX scopes, but not all native UDP/pathname-UNIX networking; the Python audit hook blocks ordinary socket use.']}

def enforce(policy,keep_fds=()):
    """Call in the fresh child before NumPy/model imports; irreversible for child."""
    abi=abi_version();lib=libc()
    thread_count=len(os.listdir('/proc/self/task'))
    if thread_count!=1: raise RuntimeError('selector isolation must precede native thread pools')
    if not policy.get('read_directories') or policy.get('run_directory_read_granted') is not False or policy.get('directory_write_create_link_granted') is not False:
        raise ValueError('invalid exact-output policy')
    # Close inherited extras before restricting; parent already uses close_fds=True.
    retained={0,1,2,*keep_fds};closed=[]
    for name in os.listdir('/proc/self/fd'):
        fd=int(name)
        if fd not in retained:
            try: os.close(fd);closed.append(fd)
            except OSError: pass
    fs=(1<<16)-1 # All FS rights through ABI5, including REFER/TRUNCATE/IOCTL_DEV.
    if abi>=9: fs|=1<<16
    net=15 if abi>=10 else 3
    attributes=Ruleset(fs,net,3)
    ruleset=lib.syscall(CREATE,ctypes.byref(attributes),ctypes.sizeof(attributes),0)
    if ruleset<0: raise OSError(ctypes.get_errno(),'Landlock ruleset creation failed')
    raw_ids={_identity(p) for p in policy['protected_paths'] if Path(p).exists()}
    grants=[]
    try:
        for names,access,is_dir in ((policy['read_directories'],READ_FILE|READ_DIR,True),
                                     (policy['read_files'],READ_FILE,False),
                                     (policy['write_files'],READ_FILE|WRITE_FILE|TRUNCATE,False)):
            for name in names:
                path=Path(name)
                if not is_dir: _regular(path)
                fd=os.open(path,os.O_PATH|os.O_CLOEXEC|os.O_NOFOLLOW)
                try:
                    st=os.fstat(fd)
                    if (st.st_dev,st.st_ino) in raw_ids: raise ValueError('raw inode at rule binding')
                    if not is_dir and (not stat.S_ISREG(st.st_mode) or st.st_nlink!=1): raise ValueError('changed exact-file inode')
                    rule=Beneath(access,fd)
                    if lib.syscall(ADD,ruleset,1,ctypes.byref(rule),0): raise OSError(ctypes.get_errno(),'Landlock rule addition failed: '+name)
                    grants.append({'path':name,'access':access,'directory':is_dir})
                finally: os.close(fd)
        if lib.prctl(NO_NEW_PRIVS,1,0,0,0): raise OSError(ctypes.get_errno(),'NO_NEW_PRIVS failed')
        flags=TSYNC if abi>=8 else 0
        if abi<8 and len(os.listdir('/proc/self/task'))!=1: raise RuntimeError('pre-ABI8 enforcement requires one thread')
        if lib.syscall(RESTRICT,ruleset,flags): raise OSError(ctypes.get_errno(),'Landlock restriction failed')
    finally: os.close(ruleset)
    return {'mechanism':'Linux Landlock','abi':abi,'no_new_privs':True,'enforced':True,
            'handled_access_fs':fs,'handled_access_net':net,'scoped':3,'restrict_flags':flags,
            'grants':grants,'policy':policy,'retained_fd_numbers':sorted(retained),'closed_extra_fds':closed,
            'before_numpy_and_model_import':True,'threads_before_restriction':thread_count}

def install_python_audit(log_fd,protected_paths):
    """Log actual CPython open attempts to preopened FD without recursive opens."""
    raw={str(Path(p).resolve()) for p in protected_paths}
    sequence=[0]
    def emit(record):
        sequence[0]+=1
        record={'sequence':sequence[0],**record}
        os.write(log_fd,(json.dumps(record,sort_keys=True)+'\n').encode())
    def hook(event,args):
        if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork','os.chdir','os.fchdir'):
            emit({'event':event,'blocked':True});raise PermissionError('selector network/spawn/directory change forbidden')
        if event=='open':
            value=args[0] if args else None
            path=None
            if isinstance(value,(str,bytes,os.PathLike)): path=str(Path(os.fsdecode(value)).resolve())
            blocked=path in raw or (path is not None and Path(path).is_relative_to('/proc'))
            emit({'event':'open','path':path,'fd_argument':value if type(value)is int else None,
                  'mode':args[1] if len(args)>1 else None,'flags':args[2] if len(args)>2 else None,'blocked':blocked})
            if blocked: raise PermissionError('protected path forbidden to selector')
    sys.addaudithook(hook)
    return emit
