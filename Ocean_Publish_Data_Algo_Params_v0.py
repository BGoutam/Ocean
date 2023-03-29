# -*- coding: utf-8 -*-
"""
Created on March 29 10:44:10 2023

@author: Goutam Bakshi, Zero2AI
"""


# Publish data NFT, datatoken, and asset for dataset based on url

# ocean.py offers multiple file object types. A simple url file is enough for here
import sys
recipientAddress = sys.argv[1]
data_URL = sys.argv[2]
dataset_name = sys.argv[3]
algo_URL = sys.argv[4]
algo_name= sys.argv[5]
p_image = sys.argv[6]
p_tag = sys.argv[7]
p_checksum = sys.argv[8]
#datafileURI = sys.argv[2]
#algofileURI = sys.argv[3]
from ocean_lib.web3_internal.utils import connect_to_network
connect_to_network("polygon-test") # mumbai is "polygon-test"
import os
from ocean_lib.example_config import get_config_dict
from ocean_lib.ocean.ocean import Ocean
#from ocean_lib.ocean.ocean_assets import OceanAssets
from ocean_lib.services.service import Service

#from ocean_lib.ocean.util import get_from_address
#from ocean_lib.assets.ddo import DDO
#from ocean_lib.models.data_nft import DataNFT
#from ocean_lib.aquarius import Aquarius
config = get_config_dict("polygon-test")
ocean = Ocean(config)
OCEAN = ocean.OCEAN_token
from brownie.network import accounts
accounts.clear()

alice_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
alice = accounts.add(alice_private_key)
assert alice.balance() > 0, "Alice needs MATIC"
assert OCEAN.balanceOf(alice) > 0, "Alice needs OCEAN"



DATA_date_created = "2019-12-28T10:55:11Z"
DATA_metadata = {
    "created": DATA_date_created,
    "updated": DATA_date_created,
    "description": "Dataset For algorithm",
    "name": "dataset",
    "type": "dataset",
    "author": "Zero2AI.io",
    "license": "MIT",
}
from ocean_lib.structures.file_objects import UrlFile
DATA_url_file = UrlFile(
    url=data_URL
)

ALGO_url_file = UrlFile(
    url=algo_URL
)


(data_nft, DATA_datatoken, DATA_ddo) = ocean.assets.create_url_asset(dataset_name, DATA_url_file.url, {"from": alice}, wait_for_aqua=True)
# Create and attach the Service

DATA_ddo.metadata = DATA_metadata

# Create and attach the Service
DATA_files = [DATA_url_file]
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
            DATA_datatoken.address,
            DATA_files,
            0,
            compute_values,
        )

DATA_ddo.add_service(compute_service)

# Update the asset
DATA_ddo = ocean.assets.update(DATA_ddo, {"from": alice})

print(f"DATA_ddo did = '{DATA_ddo.did}'")
print("Just published asset:")
print(f"  data_nft: symbol={data_nft.symbol}, address={data_nft.address}")
print(f"  datatoken: symbol={DATA_datatoken.symbol}, address={DATA_datatoken.address}")
print(f"DATA_ddo did = '{DATA_ddo.did}'")



# ocean.py offers multiple file types, but a simple url file should be enough for this example
from ocean_lib.structures.file_objects import UrlFile
ALGO_url_file = UrlFile(
    url=algo_URL
)

# Publish data NFT & datatoken for algorithm
ALGO_url = ALGO_url_file


(ALGO_data_nft, ALGO_datatoken, ALGO_ddo) = ocean.assets.create_algo_asset(algo_name, ALGO_url_file.url,  {"from": alice},image = p_image, tag = p_tag,
                                                                           checksum =p_checksum, wait_for_aqua=True)

print(f"ALGO_data_nft address = '{ALGO_data_nft.address}'")
print(f"ALGO_datatoken address = '{ALGO_datatoken.address}'")
print(f"ALGO_ddo did = '{ALGO_ddo.did}'")
# Allow C2d for the data set
compute_service = DATA_ddo.services[1]
compute_service.add_publisher_trusted_algorithm(ALGO_ddo) 
DATA_ddo = ocean.assets.update(DATA_ddo, {"from": alice})



# Publisher mints DATA datatokens and ALGO datatokens to recipient.
# Alternatively, recipient might have bought these in a market.
from ocean_lib.ocean.util import to_wei
DATA_datatoken.mint(recipientAddress, to_wei(500), {"from": alice})
ALGO_datatoken.mint(recipientAddress, to_wei(500), {"from": alice})

