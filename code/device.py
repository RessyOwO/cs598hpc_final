# ChatGPT Generated Code to check device & mem limit for later testing purposes
import pyopencl as cl

for p_idx, platform in enumerate(cl.get_platforms()):
    print(f"Platform {p_idx}: {platform.name} – {platform.vendor}")
    for d_idx, dev in enumerate(platform.get_devices()):
        print(f"  Device {d_idx}: {dev.name} ({cl.device_type.to_string(dev.type)})")
        print(f"     • Max compute units : {dev.max_compute_units}")
        print(f"     • Global mem (MiB)  : {dev.global_mem_size/1024/1024:,.0f}")
        print(f"     • Driver version    : {dev.driver_version}")
    print()
