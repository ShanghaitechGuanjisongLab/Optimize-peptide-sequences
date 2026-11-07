"""ASIC1A 的 9E4E 单链模板：生成、校验并转为 AF3 内嵌模板。

仅依赖 Python 标准库。原始 CIF 不修改；AF3 使用派生的 A 链 mmCIF，且queryIndices/templateIndices 均为SEQRES 的 0-based 下标，不是 PDB 残基号。

重新生成（会覆盖两个派生产物，不改原始 CIF）：
    python 小肽模板.py prepare
校验：
    python 小肽模板.py check
"""

from functools import lru_cache
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_CIF = os.path.join(ROOT, "Asic1a_9E4E.cif")
CHAIN_CIF = os.path.join(ROOT, "af3适配", "Asic1a_9E4E_chainA.cif")
PAYLOAD_JSON = os.path.join(ROOT, "af3适配", "Asic1a_9E4E_template.json")
PROFILE_ID = "asic1a_9e4e_chainA_seq526_v1"
CHAIN_ID = "A"


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text):
    """mmCIF 最小分词器，支持引号、注释和分号多行字符串。"""
    out, i, n, line_start = [], 0, len(text), True
    while i < n:
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            i += 1
            line_start = True
            continue
        if c == "#":
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if c == ";" and line_start:
            end = text.find("\n;", i + 1)
            if end < 0:
                raise ValueError("mmCIF 多行字符串未闭合")
            out.append(text[i + 1:end])
            i = end + 2
            line_start = False
            continue
        line_start = False
        if c in "'\"":
            j = i + 1
            while j < n:
                if text[j] == c and (j + 1 == n or text[j + 1].isspace()):
                    break
                j += 1
            if j >= n:
                raise ValueError("mmCIF 引号字符串未闭合")
            out.append(text[i + 1:j])
            i = j + 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        out.append(text[i:j])
        i = j
    return out


def _parse_cif(text):
    """返回 {类别: (字段列表, 行列表)}；同类别多个 loop 会合并。"""
    tokens = _tokens(text)
    tables, i = {}, 0
    while i < len(tokens):
        token = tokens[i]
        if token == "loop_":
            i += 1
            fields = []
            while i < len(tokens) and tokens[i].startswith("_"):
                fields.append(tokens[i])
                i += 1
            if not fields:
                raise ValueError("空 mmCIF loop")
            vals = []
            while i < len(tokens):
                # 只有完整行边界上的标记才是下一张表；PDB 的 revision_item
                # 等字段会把 _citation.xxx 作为普通数据值，不能提前截断。
                if len(vals) % len(fields) == 0 and (
                        tokens[i] in ("loop_", "stop_")
                        or tokens[i].startswith(("data_", "_"))):
                    break
                vals.append(tokens[i])
                i += 1
            if len(vals) % len(fields):
                raise ValueError(f"mmCIF loop 列数不齐：{fields[0]}")
            category = fields[0].split(".", 1)[0]
            if any(f.split(".", 1)[0] != category for f in fields):
                raise ValueError(f"暂不支持混合类别 loop：{category}")
            old_fields, rows = tables.setdefault(category, (fields, []))
            if old_fields != fields:
                raise ValueError(f"同类别 loop 字段不一致：{category}")
            rows.extend(vals[j:j + len(fields)]
                      for j in range(0, len(vals), len(fields)))
            if i < len(tokens) and tokens[i] == "stop_":
                i += 1
        elif token.startswith("_") and "." in token:
            if i + 1 >= len(tokens):
                raise ValueError(f"mmCIF 字段缺值：{token}")
            # 标量字段各自成表；同类别可能包含多个不同字段，不能按类别合并。
            tables[token] = ([token], [[tokens[i + 1]]])
            i += 2
        else:
            i += 1
    return tables


def _is_chain_field(field):
    local = field.split(".", 1)[-1].lower()
    return local in {
        "asym_id", "label_asym_id", "auth_asym_id", "pdb_strand_id",
        "pdbx_pdb_strand_id", "asym_id_list", "strand_id",
    } or local.endswith(("_label_asym_id", "_auth_asym_id"))


def _quote(value):
    if "\n" in value or len(value) > 100:
        return "\n;" + value + "\n;"
    if not value or any(c.isspace() for c in value) or value.startswith(("#", "_")):
        return "'" + value.replace("'", "''") + "'"
    return value


def _write_cif(tables, path):
    chunks = ["data_9E4E_chainA", "#"]
    for category, (fields, rows) in tables.items():
        if len(fields) == 1 and len(rows) == 1:
            chunks += [f"{fields[0]} {_quote(rows[0][0])}", "#"]
            continue
        chunks.append("loop_")
        chunks.extend(fields)
        chunks.extend(" ".join(_quote(v) for v in row) for row in rows)
        chunks.append("#")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks) + "\n")


def _template_sequence(tables):
    fields, rows = tables["_entity_poly_seq"]
    ent = fields.index("_entity_poly_seq.entity_id")
    mon = fields.index("_entity_poly_seq.mon_id")
    from 小肽数据库 import _AA3TO1
    return "".join(_AA3TO1.get(r[mon], "X") for r in rows if r[ent] == "1")


def _single_chain(text, chain_id=CHAIN_ID):
    tables = _parse_cif(text)
    for category, (fields, rows) in list(tables.items()):
        chain_cols = [j for j, field in enumerate(fields) if _is_chain_field(field)]
        kept = []
        for row in rows:
            include = True
            for j in chain_cols:
                values = row[j].split(",")
                if chain_id in values:
                    row[j] = chain_id
                elif row[j] not in (".", "?"):
                    include = False
                    break
            if include:
                kept.append(row)
        tables[category] = (fields, kept)
    for field, value in {
        "_entry.id": "9E4E_chainA",
        "_struct.entry_id": "9E4E_chainA",
        "_entity.pdbx_number_of_molecules": "1",
        "_entity_poly.pdbx_strand_id": chain_id,
    }.items():
        if field in tables:
            tables[field] = ([field], [[value]])
    seq = _template_sequence(tables)
    return tables, seq


def _align(query, template):
    """Needleman–Wunsch 全局比对，返回 [(query_idx, template_idx)]。"""
    # 分数：匹配 2、错配 -1、gap -2；长度仅 526×531，完整回溯矩阵很小。
    n, m = len(query), len(template)
    prev = [j * -2 for j in range(m + 1)]
    trace = bytearray(n * (m + 1))
    for i, aa in enumerate(query, 1):
        cur = [i * -2] + [0] * m
        for j, bb in enumerate(template, 1):
            choices = (prev[j - 1] + (2 if aa == bb else -1),
                       prev[j] - 2, cur[j - 1] - 2)
            best = max(range(3), key=lambda k: (choices[k], -k))
            cur[j], trace[(i - 1) * (m + 1) + j] = choices[best], best
        prev = cur
    pairs, i, j = [], n, m
    while i or j:
        if not i:
            j -= 1
            continue
        if not j:
            i -= 1
            continue
        op = trace[(i - 1) * (m + 1) + j]
        if op == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def prepare():
    from 小肽公共库 import ASIC1A_SEQ
    with open(SOURCE_CIF, encoding="utf-8") as f:
        source = f.read()
    tables, template_seq = _single_chain(source)
    pairs = _align(ASIC1A_SEQ, template_seq)
    matched = sum(ASIC1A_SEQ[i] == template_seq[j] for i, j in pairs)
    if len(pairs) < 500 or matched / len(pairs) < 0.90:
        raise ValueError("查询序列与 9E4E 的比对覆盖率/一致率异常，拒绝生成模板")
    os.makedirs(os.path.dirname(CHAIN_CIF), exist_ok=True)
    _write_cif(tables, CHAIN_CIF)
    with open(CHAIN_CIF, encoding="utf-8") as f:
        chain_cif = f.read()
    # 再解析派生文件，防止过滤规则留下其他链或破坏 atom_site。
    check_tables = _parse_cif(chain_cif)
    atom_fields, atoms = check_tables["_atom_site"]
    atom_chains = {r[atom_fields.index("_atom_site.label_asym_id")] for r in atoms}
    if atom_chains != {CHAIN_ID} or not atoms:
        raise ValueError(f"模板链过滤失败：{atom_chains}")
    payload = {
        "profile_id": PROFILE_ID,
        "source_cif": os.path.basename(SOURCE_CIF),
        "source_sha256": _sha256(source),
        "source_chain": CHAIN_ID,
        "template_sequence": template_seq,
        "query_sequence": ASIC1A_SEQ,
        "alignment": {
            "aligned_query_residues": len(pairs),
            "identical_residues": matched,
            "query_length": len(ASIC1A_SEQ),
            "template_length": len(template_seq),
        },
        "template": {
            "mmcif": chain_cif,
            "queryIndices": [i for i, _ in pairs],
            "templateIndices": [j for _, j in pairs],
        },
    }
    with open(PAYLOAD_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


@lru_cache(maxsize=1)
def load():
    """读取并校验不可变模板载荷；失败必须显式报错，不能静默退回无模板。"""
    from 小肽公共库 import ASIC1A_SEQ
    with open(PAYLOAD_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    with open(SOURCE_CIF, encoding="utf-8") as f:
        source = f.read()
    if payload.get("profile_id") != PROFILE_ID or payload.get("source_sha256") != _sha256(source):
        raise ValueError("9E4E 模板载荷与原始 CIF 不一致，请运行 python 小肽模板.py prepare")
    if payload.get("query_sequence") != ASIC1A_SEQ:
        raise ValueError("ASIC1A_SEQ 已改变，必须重新生成 9E4E 索引映射")
    template = payload["template"]
    qi, ti = template["queryIndices"], template["templateIndices"]
    if len(qi) != len(ti) or not qi or len(set(qi)) != len(qi) or len(set(ti)) != len(ti):
        raise ValueError("9E4E query/template 索引无效")
    if min(qi + ti) < 0 or max(qi) >= len(ASIC1A_SEQ) \
            or max(ti) >= len(payload["template_sequence"]):
        raise ValueError("9E4E 索引越界")
    return payload


def target_options(target):
    """create_af3_json/write_input_json 的靶标专属参数。"""
    if target.upper() == "ASIC1A":
        return {"target_copies": 3, "target_templates": [load()["template"]]}
    return {"target_copies": 1}


def input_matches_profile(input_path, target):
    """已有 ASIC1A 输入必须是指定 9E4E 模板；其他靶标无额外限制。"""
    if target.upper() != "ASIC1A":
        return True
    try:
        with open(input_path, encoding="utf-8") as f:
            job = json.load(f)
        proteins = [x["protein"] for x in job.get("sequences", []) if "protein" in x]
        target_protein = next(p for p in proteins if p.get("id") != ["A"])
        expected = load()["template"]
        return target_protein.get("templates") == [expected]
    except (OSError, ValueError, KeyError, StopIteration):
        return False


def main():
    import sys
    op = sys.argv[1] if len(sys.argv) > 1 else "check"
    if op == "prepare":
        payload = prepare()
    elif op == "check":
        payload = load()
    else:
        raise SystemExit("用法：python 小肽模板.py [prepare|check]")
    print(json.dumps({"profile": payload["profile_id"], **payload["alignment"],
                      "template_json": PAYLOAD_JSON}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
