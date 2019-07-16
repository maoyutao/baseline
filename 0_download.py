import hdfs
import sys
 
client = hdfs.InsecureClient("http://10.151.40.179", user="thsi_yuma")
hdfs.client._Request.webhdfs_prefix = "/webhdfs/api/v1"
 
local_file = "./"

for i in range(5):
    client.download("/Container/thsi_yuma/py_baseline/data/py{0}.pkl".format(i), local_file, overwrite=True)