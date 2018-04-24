cd /root/lichen/testDis
source activate tensorflow-3.5
python CNN1.py --job_name=ps --ps_hosts=10.1.0.113:11111 --worker_hosts=10.1.0.113:22222,10.1.0.113:22223 --task_id=0
cd /root/lichen/testDis
source activate tensorflow-3.5
python CNN1.py --job_name=worker --ps_hosts=10.1.0.113:11111 --worker_hosts=10.1.0.113:22222,10.1.0.113:22223 --task_id=0
cd /root/lichen/testDis
source activate tensorflow-3.5
python CNN1.py --job_name=worker --ps_hosts=10.1.0.113:11111 --worker_hosts=10.1.0.113:22222,10.1.0.113:22223 --task_id=1

## master: 192.168.1.117
## 一台
### --job_name=worker --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222 --task_id=0
### --job_name=ps --ps_hosts=192.168.1.117:11111 --worker_hosts=192.168.1.117:22222 --task_id=0
