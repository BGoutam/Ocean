# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:12:10 2023

@author: Goutam
"""


# Publish data NFT, datatoken, and asset for dataset based on url

# ocean.py offers multiple file object types. A simple url file is enough for here
import sys
recipientAddress = sys.argv[1]
fileURI = sys.argv[2]
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

alice_private_key = os.getenv('REMOTE_TEST_PRIVATE_KEY1')
alice = accounts.add(alice_private_key)
assert alice.balance() > 0, "Alice needs MATIC"
assert OCEAN.balanceOf(alice) > 0, "Alice needs OCEAN"


from ocean_lib.structures.file_objects import UrlFile
DATA_url_file = UrlFile(fileURI)
    #url="https://raw.githubusercontent.com/oceanprotocol/c2d-examples/main/branin_and_gpr/branin.arff"


name = "Branin dataset"
(data_nft, datatoken, ddo) = ocean.assets.create_url_asset(name, DATA_url_file.url, {"from": alice}, wait_for_aqua=True)
print("Just published asset:")
print(f"  data_nft: symbol={data_nft.symbol}, address={data_nft.address}")
print(f"  datatoken: symbol={datatoken.symbol}, address={datatoken.address}")
print(f"  did={ddo.did}")


# NOTE :: Minting datatoken for bob's use
#   bob's address is hard coded. When making command line
# This should be passed as args. Also, multiple addresses can be issued
# data tokens.
#bob = "0xDF9bC869CC5E81887E95Fb3B83861A372408CA6F"
datatoken.mint(recipientAddress, to_wei(1), {"from": alice})

