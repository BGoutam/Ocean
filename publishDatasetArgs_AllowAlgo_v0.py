# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:12:10 2023

@author: Goutam bakshi, Zero2AI
"""


# Publish data NFT, datatoken, and asset for dataset based on url

# ocean.py offers multiple file object types. A simple url file is enough for here
import sys
recipientAddress = sys.argv[1]
fileURI = sys.argv[2]
dataset_name = sys.argv[3]
allowed_algo_DID = sys.argv[4]
from ocean_lib.web3_internal.utils import connect_to_network
connect_to_network("polygon-test") # mumbai is "polygon-test"
import os
from ocean_lib.example_config import get_config_dict
from ocean_lib.services.service import Service
from ocean_lib.ocean.ocean import Ocean
from ocean_lib.ocean.util import to_wei
config = get_config_dict("polygon-test")
ocean = Ocean(config)
OCEAN = ocean.OCEAN_token
from brownie.network import accounts
accounts.clear()

alice_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
alice = accounts.add(alice_private_key)
assert alice.balance() > 0, "Alice needs MATIC"
assert OCEAN.balanceOf(alice) > 0, "Alice needs OCEAN"


from ocean_lib.structures.file_objects import UrlFile
#Need to attach a compute service for this
# Create and attach the Service

DATA_url_file = UrlFile(fileURI)


name = dataset_name
(data_nft, datatoken, ddo) = ocean.assets.create_url_asset(name, DATA_url_file.url, {"from": alice}, wait_for_aqua=True)

#Need to add compute service to the dataset and allow it to be
#executed by the specified algorithm
DATA_files = [DATA_url_file]
DATA_ddo = ddo
compute_values = {
                "allowRawAlgorithm": False,
                "allowNetworkAccess": True,
                "publisherTrustedAlgorithms": [],
                "publisherTrustedAlgorithmPublishers": [],
            }

compute_service = Service(
            "2",
            "compute",
            ocean.config_dict["PROVIDER_URL"],
            datatoken.address,
            DATA_files,
            0,
            compute_values,
        )

ddo.add_service(compute_service)

ddo = ocean.assets.update(ddo, {"from": alice})
# NOTE :: Minting datatoken for bob's use
#   bob's address is hard coded. When making command line
# This should be passed as args. Also, multiple addresses can be issued
# data tokens.
#bob = "0xDF9bC869CC5E81887E95Fb3B83861A372408CA6F"
datatoken.mint(recipientAddress, to_wei(10), {"from": alice})
# Allowing Algorithm


ALGO_ddo = ocean.assets.resolve(allowed_algo_DID)
added_service = ddo.services[1]
added_service.add_publisher_trusted_algorithm(ALGO_ddo) 
ddo = ocean.assets.update(ddo, {"from": alice})

print("Just published asset:")
print(f"  data_nft: symbol={data_nft.symbol}, address={data_nft.address}")
print(f"  datatoken: symbol={datatoken.symbol}, address={datatoken.address}")
print(f"  did={ddo.did}")


