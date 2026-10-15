# 文件作用：依据已有验证证据生成实现审计；此模块本身不会执行服务器验收。
"""Evidence-bound implementation audit."""
import hashlib,json,platform,time
from pathlib import Path

# 读取证据、区分本地/WSL/服务器验证状态，并写出禁止生产发布的审计报告。
def run(output,evidence=None):
    evidence_path=Path(evidence) if evidence else Path(output).with_name("final-audit.json")
    ev=json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    evidence_platform = str(ev.get("environment",{}).get("platform", ""))
    evidence_is_linux = "linux" in evidence_platform.lower() or "wsl" in evidence_platform.lower()
    verified = "verified_in_wsl" if evidence_is_linux else "implemented_local_only"
    checks={
      "association_lifecycle":[verified if ev.get("checks",{}).get("lifecycle",{}).get("status")=="passed" else "missing","prepared/waiting reducer evidence"],
      "canonical_features":[verified if ev.get("checks",{}).get("canonical_arguments",{}).get("status")=="passed" else "not_verified","equivalent JSON arguments"],
      "sentence_transformer":[verified if ev.get("checks",{}).get("sentence_transformer",{}).get("status")=="passed" else "not_verified","real encoder is required; no hashing fallback"],
      "redis_cas_streams":["verified_in_wsl" if ev.get("checks",{}).get("redis_real",{}).get("status")=="passed" else "not_verified","real Redis Streams/CAS evidence"],
      "sglang_gpu_protocol":["not_verified_server","requires pinned SGLang GPU service"],
      "formal_return_labels":["implemented","event-backed exact/censored schema; real observations still require server evidence"],
      "gbdt_regression_contract":["implemented","structured normal-AFT baseline; joint-semantic ablation is optional"],
      "plugin_api":["implemented","versioned window/distribution/lease contract and scheduler-neutral consumer"],
      "batch_retraining":["implemented","immutable event cutoff, renewed training lease and quality-gated promotion"],
      "accuracy_curve":["implemented","per-round P50/P90 relative-error hit rates"],
      "atomic_package":["implemented","file fsync and atomic best-model link"],
      "server_performance":["not_verified_server","P99 and throughput require target hosts"]}
    result={"status":"incomplete" if any(v[0] in ("missing","not_verified_server","not_verified") for v in checks.values()) else "complete","generated_at":time.time(),"environment":ev.get("environment",{"platform":platform.platform(),"python":platform.python_version()}),"source_digest":ev.get("source_digest") or ev.get("checks",{}).get("source_digest",{}).get("sha256"),"checks":{k:{"status":v[0],"detail":v[1]} for k,v in checks.items()},"production_publish_allowed":False}
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); return result
