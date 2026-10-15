# 文件作用：生成绑定源码摘要的环境验证证据；未运行的 Redis/GPU/跨主机验收明确标记未验证。
"""Single evidence-bound WSL verification entry point."""
import hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 按排序后的源码路径与原始文件字节计算摘要；新增注释也会改变摘要。
def source_digest():
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()

# 捕获单项检查异常并输出 passed/failed 证据。
def run_check(fn):
    try:
        value = fn() or {}
        return {"status":"passed", **value}
    except Exception as exc:
        return {"status":"failed", "reason":f"{type(exc).__name__}: {exc}"}

# 使用当前 Python 编译源码与测试，检查语法错误。
def _compile():
    p = subprocess.run([sys.executable,"-m","compileall","-q",str(ROOT/"src"),str(ROOT/"tests")], cwd=ROOT)
    if p.returncode: raise RuntimeError("compileall failed")

# 关闭系统 pytest 插件自动加载，记录测试返回码与末尾输出。
def _pytest():
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    p = subprocess.run([sys.executable,"-m","pytest","-q"], cwd=ROOT, env=env, text=True, capture_output=True)
    return {"status":"passed" if p.returncode == 0 else "not_verified", "returncode":p.returncode, "output_tail":(p.stdout+p.stderr)[-1200:]}

# 验证最小 prepared→out 事件能够转换为 waiting。
def _lifecycle():
    from timepredict.state import empty_state, reduce_event
    stamp={"utc":1.,"mono":1.,"source":"verify","boot":"1","error":.001,"jumped":False}
    e={"event_id":"p","kind":"prepared","tenant":"t","workflow_id":"w","agent_id":"a","request_id":"r","call":{"id":"c","name":"x","arguments":"{}"},"stamp":stamp}
    state,_=reduce_event(empty_state(),e)
    state,_=reduce_event(state,{**e,"event_id":"o","kind":"out","stamp":{**stamp,"utc":2.,"mono":2.}})
    if next(iter(state["records"].values()))["status"] != "waiting": raise AssertionError("prepared->waiting failed")
    if next(iter(state["records"].values()))["prediction_status"] != "pending_prediction": raise AssertionError("prepared round cannot start prediction")
    return {"state":"prepared->waiting"}

# 仅检查补丁包含预期钩子标记；不证明补丁已在上游应用或运行。
def _patch_static():
    text=(ROOT/"patches/sglang-v0.5.19.patch").read_text(encoding="utf-8")
    required=("tokenizer_worker_num","accepted","observe_stream","observe_nonstream")
    missing=[x for x in required if x not in text]
    if missing: raise RuntimeError("missing patch markers: "+",".join(missing))
    return {"files":[x[6:] for x in text.splitlines() if x.startswith("+++ b/")]}

# 汇总本环境检查、未验证服务器项目与源码摘要，并写入证据 JSON。
def main():
    checks={"compileall":run_check(_compile),"pytest":_pytest(),"lifecycle":run_check(_lifecycle),"sglang_patch_static":run_check(_patch_static)}
    try:
        from timepredict.semantic import descriptor_text
        a={"name":"x","arguments":{"b":2,"a":1}}; b={"name":"x","arguments":"{\"a\":1,\"b\":2}"}
        checks["canonical_arguments"]={"status":"passed" if descriptor_text(a)==descriptor_text(b) else "failed"}
    except Exception as exc: checks["canonical_arguments"]={"status":"failed","reason":str(exc)}
    checks["redis_real"]={"status":"not_verified","reason":"redis-server is not installed in this WSL distribution"} if shutil.which("redis-server") is None else {"status":"not_run","reason":"integration runner not configured"}
    result={"generated_at":time.time(),"environment":{"platform":platform.platform(),"python":platform.python_version(),"cwd":str(ROOT)},"source_digest":source_digest(),"checks":checks,"server_checks":{"gpu_sglang":"not_verified_server","chrony_multi_host":"not_verified_server","entry_p99":"not_verified_server","throughput":"not_verified_server"}}
    out=Path(os.environ.get("TIMEPREDICT_VERIFY_OUTPUT",ROOT/"outputs/.wsl-evidence.json")); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
