"""Deterministic semantic action-family matching without exact-name joins."""
import re
import math
from collections import Counter
from dataclasses import asdict, dataclass, field

from .semantic import SemanticEncoder, descriptor_text

STOP = {"the", "a", "an", "to", "of", "for", "with", "and", "tool", "call", "function"}
SYNONYMS = {"find": "search", "lookup": "search", "query": "search", "retrieve": "search",
            "remove": "delete", "erase": "delete", "make": "create", "run": "execute",
            "write": "create", "fetch": "search", "get": "read"}


def _tokens(value):
    words = re.findall(r"[a-z0-9]+", str(value or "").lower().replace("_", " ").replace("-", " "))
    return {SYNONYMS.get(word, word) for word in words if word not in STOP and len(word) > 1}


def _descriptor(call):
    schema = call.get("schema") if isinstance(call.get("schema"), dict) else {}
    props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
    enums = []
    types = []
    for key, value in props.items():
        types.append(f"{key}:{value.get('type', 'unknown') if isinstance(value, dict) else 'unknown'}")
        if isinstance(value, dict):
            enums.extend(map(str, value.get("enum", [])))
    args = call.get("arguments", {})
    if isinstance(args, str):
        import json
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("arguments_not_object")
    arg_fields = sorted(args) if isinstance(args, dict) else []
    arg_types = sorted(f"{key}:{type(value).__name__}" for key, value in args.items()) if isinstance(args, dict) else []
    # Family identity describes an operation contract. Concrete argument
    # values belong in numeric/string-length features, otherwise every unique
    # query becomes a new sparse family and cannot generalize to the test set.
    text = " ".join((str(call.get("name", "")), str(call.get("description", "")),
                     " ".join(arg_fields + arg_types), " ".join(enums), str(call.get("role", ""))))
    return {"tokens": sorted(_tokens(text)), "types": sorted(set(types + arg_types)),
            "fields": sorted(set(list(props) + arg_fields)), "role": call.get("role", ""), "text": text}


def _signature(descriptor):
    return (tuple(descriptor["tokens"]), tuple(descriptor["types"]), tuple(descriptor["fields"]), descriptor["role"])


def _score(left, right, idf=None):
    a, b = set(left["tokens"]), set(right["tokens"])
    token = len(a & b) / len(a | b) if a | b else 0.0
    af, bf = set(left["fields"]), set(right["fields"])
    fields = len(af & bf) / len(af | bf) if af | bf else 0.0
    at, bt = set(left["types"]), set(right["types"])
    types = len(at & bt) / len(at | bt) if at | bt else 0.0
    role = 1.0 if left["role"] and left["role"] == right["role"] else 0.0
    def ngrams(value, n=3):
        value = re.sub(r"\s+", " ", value.lower())
        return {value[i:i+n] for i in range(max(0, len(value)-n+1))}
    na, nb = ngrams(left["text"]), ngrams(right["text"])
    ngram = len(na & nb) / len(na | nb) if na | nb else 0.0
    # A compact TF-IDF-like cosine over descriptor tokens. IDF is applied by
    # the family representatives at match time, without a cross-dataset join.
    ca, cb = Counter(left["tokens"]), Counter(right["tokens"])
    idf = idf or {}
    common = set(ca) & set(cb)
    def weight(counter, token):
        return counter[token] * idf.get(token, 1.0)
    cosine = sum(weight(ca, t) * weight(cb, t) for t in common)
    cosine /= math.sqrt(sum(weight(ca, t)**2 for t in ca) * sum(weight(cb, t)**2 for t in cb)) if ca and cb else 1.0
    semantic = float(left.get("embedding_score", 0.0))
    return 0.25 * token + 0.15 * ngram + 0.20 * fields + 0.12 * types + 0.05 * role + 0.13 * cosine + 0.10 * semantic


@dataclass
class ActionFamilies:
    representatives: list
    threshold: float = 0.42
    margin: float = 0.08
    version: int = 2
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_dim: int = 384
    embedding_kind: str = "sentence_transformers"
    idf: dict = field(default_factory=dict)
    _encoder: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self._encoder is None:
            try:
                self._encoder = SemanticEncoder(self.embedding_model, revision=self.embedding_revision)
                if self._encoder.dim != self.embedding_dim or self._encoder.backend != self.embedding_kind:
                    raise ValueError("embedding_contract_mismatch")
            except Exception:
                raise

    @classmethod
    def fit(cls, calls):
        reps, assignments = [], []
        encoder = SemanticEncoder()
        for call in calls:
            descriptor = _descriptor(call)
            vector = encoder.encode(call)
            scores = [_score({**descriptor, "embedding_score": float(vector @ item["embedding"])}, item["descriptor"], cls._idf(reps)) for item in reps]
            same = next((item for item in reps if _signature(item["descriptor"]) == _signature(descriptor)), None)
            if same is not None:
                same["members"].append(call.get("name", ""))
            elif scores and max(scores) >= 0.42 and (len(scores) == 1 or max(scores) - sorted(scores)[-2] >= 0.08):
                index = max(range(len(scores)), key=scores.__getitem__)
                reps[index]["members"].append(call.get("name", ""))
            else:
                reps.append({"family": f"action_{len(reps):04d}", "descriptor": descriptor,
                             "embedding": vector.tolist(), "members": [call.get("name", "")]})
        return cls(reps, embedding_model=encoder.model_name, embedding_revision=encoder.revision,
                   embedding_dim=encoder.dim, embedding_kind=encoder.backend, idf=cls._idf(reps))

    @staticmethod
    def _idf(representatives):
        documents = [set(item["descriptor"]["tokens"]) for item in representatives]
        total = max(1, len(documents))
        return {token: math.log((1 + total) / (1 + sum(token in doc for doc in documents))) + 1.0
                for document in documents for token in document}

    def match(self, call):
        descriptor = _descriptor(call)
        vector = self._encoder.encode(call) if self._encoder else None
        def similarity(item):
            stored = item.get("embedding", [])
            if vector is None or not stored:
                score = 0.0
            else:
                if len(vector) != len(stored):
                    raise ValueError("embedding_dimension_mismatch")
                score = float(vector @ stored)
            return float(_score({**descriptor, "embedding_score": score}, item["descriptor"], self.idf))
        ranked = sorted(((similarity(item), item["family"]) for item in self.representatives), reverse=True)
        if not ranked or ranked[0][0] < self.threshold:
            return "unknown_action", ranked[:3], descriptor
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.margin:
            return "ambiguous_action", ranked[:3], descriptor
        return ranked[0][1], ranked[:3], descriptor

    def enrich(self, call):
        family, ranked, descriptor = self.match(call)
        result = dict(call)
        result["action_family"] = family
        result["action_match"] = {"family": family, "ranked": ranked, "descriptor": descriptor}
        return result

    def to_dict(self):
        return {"representatives": self.representatives, "threshold": self.threshold,
                "margin": self.margin, "version": self.version,
                "embedding_model": self.embedding_model, "embedding_dim": self.embedding_dim,
                "embedding_revision": self.embedding_revision,
                "embedding_kind": self.embedding_kind, "idf": self.idf}
