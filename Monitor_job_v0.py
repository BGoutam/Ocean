# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:44:42 2023

@author: Goutam Bakshi, Zero2AI
"""
from ocean_lib.web3_internal.utils import connect_to_network
connect_to_network("polygon-test") # mumbai is "polygon-test"
import os, sys

dataDID = sys.argv[1]
job_id = sys.argv[2]
dumpFile = sys.argv[3]
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean

config = get_config_dict("polygon-test")
ocean = Ocean(config)
OCEAN = ocean.OCEAN_token
from brownie.network import accounts
accounts.clear()

bob_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY2')
bob = accounts.add(bob_private_key)
assert bob.balance() > 0, "Bob needs MATIC"
assert OCEAN.balanceOf(bob) > 0, "Bob needs OCEAN"

DATA_ddo = ocean.assets.resolve(dataDID)

compute_service = DATA_ddo.services[1]

import time
from decimal import Decimal
succeeded = False
print("Monitoring Starts..")
for _ in range(0, 200000):
    status = ocean.compute.status(DATA_ddo, compute_service, job_id, bob)
    print("Status of Job ",status)
    if status.get("dateFinished") and Decimal(status["dateFinished"]) > 0:
        print("Job Finished")
        succeeded = True
        break
    time.sleep(5)
result =ocean.compute.result(DATA_ddo, compute_service, job_id,1,bob)
print("result)") 
print(result)
output = ocean.compute.compute_job_result_logs(
    DATA_ddo, compute_service, job_id, bob
)
print("Completed Execution")
num_out = len(output)
print("number of outputs",num_out)
for i in range(len(output)):
    print("output Type",type(output[i]))
    dfile = dumpFile+'_'+ str(i)

    with open(dfile, "wb") as dump_file:
        dump_file.write(output[i])
        dump_file.close()




