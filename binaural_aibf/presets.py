"""模型版本与探针档位预设（纯函数，无文件 IO，不修改入参）。"""

import copy

BANDWIDTH_PRESETS = (
    "local",
    "float",
    "complex_2bit",
    "default",
    "complex_6bit",
    "slow",
    "magnitude",
    "matched_rate_magnitude",
)

VERSION_PRESETS = (
    "local",
    "float",
    "s0_default",
    "v1_mixed",
)


def preset(cfg: dict, name: str) -> dict:
    """按档位名返回深拷贝并覆盖后的 7 小节配置字典。

    冻结支持：local、float、complex_2bit、default、complex_6bit、slow、
    magnitude、matched_rate_magnitude、delay_N，以及 s0_default、v1_mixed。
    """
    c = copy.deepcopy(cfg)
    m = c["message"]
    m["enabled"] = name != "local"

    if name == "float":
        n_bands = int(c["bands"]["n_linear"]) + int(c["bands"]["n_tri"])
        m.update(
            payload="complex",
            band_range=[62.5, 8000.0],
            n_msg_bands=n_bands,
            bits_mag=0,
            bits_phase=0,
            update_period=1,
            delay_frames=0,
        )
    elif name in (
        "default",
        "complex_2bit",
        "complex_6bit",
        "slow",
        "magnitude",
        "matched_rate_magnitude",
    ):
        if m.get("payload") == "mixed":
            m.update(
                payload="complex",
                band_range=[125.0, 1000.0],
                n_msg_bands=8,
                bits_mag=4,
                bits_phase=4,
            )
        if name in ("complex_2bit", "complex_6bit"):
            bits = 2 if name == "complex_2bit" else 6
            m.update(payload="complex", bits_mag=bits, bits_phase=bits)
        elif name == "slow":
            m["update_period"] = 8
        elif name == "magnitude":
            m.update(payload="logmag", bits_phase=0)
        elif name == "matched_rate_magnitude":
            m.update(
                payload="logmag",
                bits_phase=0,
                bits_mag=int(m["bits_mag"]) + int(m["bits_phase"]),
            )
    elif name.startswith("delay_"):
        m["delay_frames"] = int(name.split("_", 1)[1])
    elif name == "s0_default":
        m.update(
            payload="complex",
            band_range=[125.0, 1000.0],
            n_msg_bands=8,
            bits_mag=4,
            bits_phase=4,
            update_period=4,
            delay_frames=4,
        )
        c["crf"]["use_frontal_features"] = False
        c["mvdr"].update(hidden=64, use_film=False, use_cross_band=False)
    elif name == "v1_mixed":
        m.update(
            payload="mixed",
            bits_ref=6,
            ref_range_db=[-60.0, 0.0],
            phase_band_range=[125.0, 1375.0],
            n_phase_bands=6,
            bits_phase=5,
            mag_band_range=[1500.0, 7700.0],
            n_mag_bands=7,
            bits_mag=4,
            rel_mag_range_db=[-24.0, 12.0],
            update_period=4,
            delay_frames=4,
        )
        c["crf"]["use_frontal_features"] = True
        c["mvdr"].update(hidden=48, use_film=True, use_cross_band=True)
    elif name != "local":
        raise ValueError(f"未知配置档位 {name}")
    return c
