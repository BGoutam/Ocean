# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 00:27:01 2023

@author: Goutam
"""
import sys
dataDID = sys.argv[1]
from ocean_lib.web3_internal.utils import connect_to_network
connect_to_network("polygon-test") # mumbai is "polygon-test"
import os
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.ocean.util import to_wei
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
did = dataDID
ddo = ocean.assets.resolve(did)
# Bob sends a datatoken to the service to get access
order_tx_id = ocean.assets.pay_for_access_service(ddo, {"from": bob})
print("Transaction Order ID:",order_tx_id)

# Bob downloads the file. If the connection breaks, Bob can try again
asset_dir = ocean.assets.download_asset(ddo, bob, './', order_tx_id)

import os
file_name = os.path.join(asset_dir, "file0")
print("File Name:" ,file_name)