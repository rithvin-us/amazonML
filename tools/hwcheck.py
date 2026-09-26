import psutil, shutil
vm = psutil.virtual_memory(); sw = psutil.swap_memory()
print('RAM total %.1f avail %.1f GB used %d%%' % (vm.total/2**30, vm.available/2**30, vm.percent))
print('pagefile %.1f used %.1f GB' % (sw.total/2**30, sw.used/2**30))
for d in ('C:/', 'D:/'):
    u = shutil.disk_usage(d); print(d, 'free %.1f GB' % (u.free/2**30))
ps = sorted(psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']),
            key=lambda p: -(p.info['memory_info'].rss if p.info['memory_info'] else 0))[:12]
for p in ps:
    cl = ' '.join(p.info['cmdline'] or [])[-70:] if p.info['name'] == 'python.exe' else ''
    print(p.info['pid'], p.info['name'], round(p.info['memory_info'].rss/2**20), 'MB', cl)
