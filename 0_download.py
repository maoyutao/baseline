import hdfs
import sys
 
client = hdfs.InsecureClient("http://10.151.40.179", user="thsi_yuma")
hdfs.client._Request.webhdfs_prefix = "/webhdfs/api/v1"
 
pai_file = "/Container/thsi_yuma/py_baseline/data/py.csv"
local_file = "./"
client.download(pai_file, local_file, overwrite=True)