# 文件作用：冻结语义编码器与操作描述文本；生产仅使用本地固定模型，测试可显式注入哈希替身。
"""Frozen semantic representation shared by fitting and serving."""
from collections import OrderedDict
import json
import os
import re
from typing import Any
import numpy as np
VERSION=3
DEFAULT_MODEL="sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION=os.environ.get("TIMEPREDICT_EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
STOP={"the","a","an","to","of","for","with","and","tool","call","function"}
SYNONYMS={"find":"search","lookup":"search","query":"search","retrieve":"search","fetch":"search","remove":"delete","erase":"delete","make":"create","write":"create","run":"execute","invoke":"execute","list":"read"}
# 兼容 JSON 字符串和字典，拒绝非对象参数。
def _canonical_arguments(value: Any)->dict:
    if isinstance(value,str): value=json.loads(value)
    if not isinstance(value,dict): raise ValueError("arguments_not_object")
    return value
# 将 schema 字段、类型和枚举规范化为文本。
def _schema_text(schema):
    parts=[]
    # 按当前函数的输入递归提取 schema 或参数结构；字段顺序固定。
    def walk(node,prefix=""):
        if not isinstance(node,dict): return
        if prefix: parts.append(prefix)
        if node.get("type"): parts.append(str(node["type"]))
        parts.extend(str(x) for x in node.get("enum",[]) if isinstance(node.get("enum"),list))
        props=node.get("properties",{})
        for key in sorted(props) if isinstance(props,dict) else []: walk(props[key],key)
    walk(schema or {})
    return " ".join(parts)
# 描述操作契约与参数结构，排除实际参数值，避免合成延时公式泄漏到语义向量。
def descriptor_text(call):
    args=_canonical_arguments(call.get("arguments",{}))
    # Describe the operation contract without embedding observed values.
    # Values are represented by explicit structured features; including them in
    # the language vector would create a shortcut to synthetic delay formulas.
    structure=[]
    # 按当前函数的输入递归提取 schema 或参数结构；字段顺序固定。
    def walk(value, path=""):
        if isinstance(value, dict):
            for key in sorted(value):
                child=value[key]
                structure.append(f"{path}/{key}:{type(child).__name__}")
                walk(child, f"{path}/{key}")
        elif isinstance(value, list):
            structure.append(f"{path}:array")
            if value:
                walk(value[0], path+"[]")
    walk(args)
    return " ".join((str(call.get("name","")),str(call.get("description","")),
                     _schema_text(call.get("schema",{})),str(call.get("role","")),
                     " ".join(structure)))
# 测试专用确定性哈希编码器，不能充当生产语义模型。
class FrozenSemanticEncoder:
    """Test-only deterministic encoder; never selected for production."""
    # 初始化编码器；真实后端必须符合固定模型契约，测试替身只由显式环境变量启用。
    def __init__(self,dim=32,version=VERSION):
        import hashlib
        if int(dim)<=0 or int(version)!=VERSION: raise ValueError("embedding_contract_mismatch")
        self.dim,self.version,self._hashlib=int(dim),int(version),hashlib
    # 将调用的操作描述编码为向量；实际算法由所在编码器类决定。
    def encode(self,call):
        vector=np.zeros(self.dim,dtype=np.float32)
        words=[SYNONYMS.get(w,w) for w in re.findall(r"[a-z0-9]+",descriptor_text(call).lower()) if w not in STOP]
        for feature in words+[words[i]+"::"+words[i+1] for i in range(len(words)-1)]:
            digest=self._hashlib.blake2b((f"{self.version}:{feature}").encode(),digest_size=8).digest(); vector[int.from_bytes(digest[:4],"little")%self.dim]+=1 if digest[4]&1 else -1
        norm=np.linalg.norm(vector); return vector/norm if norm else vector
    # 导出编码器后端、维度和版本契约。
    def to_dict(self): return {"kind":"test_hashing","version":self.version,"dim":self.dim}
# 复用固定模型实例和文本向量 LRU 缓存；返回向量副本防止调用方污染缓存。
class SemanticEncoder:
    _instances = OrderedDict()
    # 按模型、缓存目录、revision 与设备查找已初始化实例。
    def __new__(cls, model_name=DEFAULT_MODEL, cache_folder=None, revision=DEFAULT_REVISION, device="cpu", max_cache=2048):
        key = (model_name, cache_folder or os.environ.get("TIMEPREDICT_EMBEDDING_CACHE"), revision, device)
        cached = cls._instances.get(key)
        if cached is not None:
            cls._instances.move_to_end(key)
            return cached
        obj = super().__new__(cls)
        obj._cache_key = key
        return obj
    # 初始化编码器；真实后端必须符合固定模型契约，测试替身只由显式环境变量启用。
    def __init__(self,model_name=DEFAULT_MODEL,cache_folder=None,revision=DEFAULT_REVISION,device="cpu",max_cache=2048):
        cache_key = self._cache_key
        if getattr(self, "_initialized", False):
            return
        if model_name!=DEFAULT_MODEL: raise ValueError("unsupported_embedding_model")
        self.model_name,self.revision,self.device=model_name,revision,device
        self.cache_folder=cache_folder or os.environ.get("TIMEPREDICT_EMBEDDING_CACHE")
        self.backend="sentence_transformers"
        if os.environ.get("TIMEPREDICT_TEST_ENCODER") == "1":
            self.backend = "test_hashing"; self.model = FrozenSemanticEncoder(384); self.dim = 384; self.version = VERSION; self._cache = OrderedDict(); self._max_cache = max(1, int(max_cache)); self._initialized = True; self._instances[cache_key] = self; return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            if os.environ.get("TIMEPREDICT_TEST_ENCODER") == "1":
                self.backend = "test_hashing"; self.model = FrozenSemanticEncoder(384); self.dim = 384; self.version = VERSION; self._cache = OrderedDict(); self._max_cache = max(1, int(max_cache)); self._initialized=True; self._instances[cache_key]=self; return
            raise RuntimeError("sentence_transformers_unavailable") from exc
        try: self.model=SentenceTransformer(model_name,revision=revision,cache_folder=self.cache_folder,device=device,local_files_only=True)
        except TypeError: self.model=SentenceTransformer(model_name,revision=revision,cache_folder=self.cache_folder,device=device,model_kwargs={"local_files_only":True})
        self.dim,self.version=int(self.model.get_sentence_embedding_dimension()),VERSION
        self._cache,self._max_cache=OrderedDict(),max(1,int(max_cache))
        self._initialized = True
        self._instances[cache_key] = self
        while len(self._instances) > 4: self._instances.popitem(last=False)
    # 按 tokenizer 与模型容量分块，预留特殊 token，避免长描述被静默截断。
    def _chunks(self,text):
        tokenizer=getattr(self.model,"tokenizer",None)
        if tokenizer is None: return [text]
        ids=tokenizer(text,add_special_tokens=False,truncation=False,return_attention_mask=False).get("input_ids",[])
        limit=int(getattr(tokenizer,"model_max_length",256)); limit=256 if limit<=0 or limit>100000 else limit
        model_limit = getattr(self.model, "max_seq_length", None)
        if model_limit: limit = min(limit, int(model_limit))
        limit = max(8, limit - 2)  # reserve special tokens added by the encoder
        return [tokenizer.decode(ids[i:i+limit],skip_special_tokens=True) for i in range(0,len(ids),limit)] or [""]
    # 编码每段文本、取均值并 L2 归一化，缓存结果；测试分支使用哈希替身。
    def encode_text(self,text):
        key=str(text)
        if key in self._cache:
            value=self._cache.pop(key); self._cache[key]=value; return value.copy()
        # 此分支把文本放进 arguments.text；测试编码器只看参数结构，因此当前不同文本可能得到相同向量。
        if self.backend == "test_hashing":
            vector = self.model.encode({"name": "", "arguments": {"text": key}})
            self._cache[key] = vector
            return vector.copy()
        vectors=self.model.encode(self._chunks(key),normalize_embeddings=False,convert_to_numpy=True,device=self.device)
        vector=np.asarray(vectors,dtype=np.float32).mean(axis=0); norm=float(np.linalg.norm(vector)); vector=vector/norm if norm else vector
        self._cache[key]=vector
        while len(self._cache)>self._max_cache: self._cache.popitem(last=False)
        return vector.copy()
    # 将调用的操作描述编码为向量；实际算法由所在编码器类决定。
    def encode(self,call): return self.encode_text(descriptor_text(call))
    # 导出编码器后端、维度和版本契约。
    def to_dict(self): return {"kind":self.backend,"model":self.model_name,"revision":self.revision,"dim":self.dim,"version":self.version,"device":self.device,"chunking":"tokenizer_capacity_mean_then_l2_normalize"}
