"""Pure map-payload lowering shared by legacy and function instrumentation."""


def emit_save(map_, items, pred="", *, vectorize=True):
    """Store one map record and advance its per-sample pointer.

    Lane predicates are supplied by the backend. Vector stores require alignment
    at every record, including odd-sized records and mixed-width field tails.
    """
    if len(items) != len(map_.regs):
        raise ValueError(f"{map_.name}.save does not follow its field definition")
    cvt_lines, u64s, u32s = [], [], []
    u32_idx = 1
    for item, reg in zip(items, map_.regs):
        if reg.dtype == "u64":
            u64s.append(item)
        elif reg.dtype == "u32":
            cvt_lines.append(f"{pred}cvt.u32.u64 %{map_.name}_u32_{u32_idx}, {item};\n")
            u32s.append(f"%{map_.name}_u32_{u32_idx}")
            u32_idx += 1
        else:
            raise ValueError(f"Unsupported map field type: {reg.dtype}")
    if len(u32s) % 2:
        raise ValueError("Map records require an even number of u32 fields")
    save_lines = []
    if not vectorize or map_.size % 16 or (len(u64s) % 2 and len(u32s) >= 4):
        for dtype, values, size in (("u64", u64s, 8), ("u32", u32s, 4)):
            for value in values:
                save_lines.append(
                    f"{pred}st.global.{dtype} [%map_{map_.name}1], {value};\n"
                    f"{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, {size};")
    else:
        for item_idx in range(len(u64s) // 2):
            save_lines.append(f"{pred}st.global.v2.u64 [%map_{map_.name}1],  {{ {u64s[item_idx * 2]}, {u64s[item_idx * 2 + 1]} }};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 16;")
        if len(u64s) % 2:
            save_lines.append(f"{pred}st.global.u64 [%map_{map_.name}1], {u64s[-1]};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 8;")
        for item_idx in range(len(u32s) // 4):
            save_lines.append(f"{pred}st.global.v4.u32 [%map_{map_.name}1],  {{ {u32s[item_idx * 4]}, {u32s[item_idx * 4 + 1]}, {u32s[item_idx * 4 + 2]}, {u32s[item_idx * 4 + 3]} }};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 16;")
        if len(u32s) % 4:
            save_lines.append(f"{pred}st.global.v2.u32 [%map_{map_.name}1],  {{ {u32s[-2]}, {u32s[-1]} }};\n{pred}add.s64 %map_{map_.name}1, %map_{map_.name}1, 8;")
    return "\n".join(cvt_lines) + "\n".join(save_lines)
