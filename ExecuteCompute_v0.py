# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 00:27:01 2023

@author: Goutam bakshi, Zero2AI
"""
import sys
dataDID = sys.argv[1]
algoDID = sys.argv[2]
#dumpFile = sys.argv[3]
from ocean_lib.web3_internal.utils import connect_to_network
connect_to_network("polygon-test") # mumbai is "polygon-test"
import os
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
# Note below the did is the one received as output of 
# publisg data set
# Operate on updated and indexed assets
DATA_ddo = ocean.assets.resolve(dataDID)
ALGO_ddo = ocean.assets.resolve(algoDID)

compute_service = DATA_ddo.services[1]
algo_service = ALGO_ddo.services[0]
free_c2d_env = ocean.compute.get_free_c2d_environment(compute_service.service_endpoint)

from datetime import datetime, timedelta
from ocean_lib.models.compute_input import ComputeInput

DATA_compute_input = ComputeInput(DATA_ddo, compute_service)
ALGO_compute_input = ComputeInput(ALGO_ddo, algo_service)

# Pay for dataset and algo for 1 day
datasets, algorithm = ocean.assets.pay_for_compute_service(
    datasets=[DATA_compute_input],
    algorithm_data=ALGO_compute_input,
    consume_market_order_fee_address=bob.address,
    tx_dict={"from": bob},
    compute_environment=free_c2d_env["id"],
    valid_until=int((datetime.utcnow() + timedelta(days=1)).timestamp()),
    consumer_address=free_c2d_env["consumerAddress"],
)
assert datasets, "pay for dataset unsuccessful"
assert algorithm, "pay for algorithm unsuccessful"

# Start compute job
job_id = ocean.compute.start(
    consumer_wallet=bob,
    dataset=datasets[0],
    compute_environment=free_c2d_env["id"],
    algorithm=algorithm,
)
print(f"Started compute job with id: {job_id}")


