#!/bin/bash
cd /home/nsuser/ns-3.48
IN=/mnt/f/leo-routing-preliminary-matlab/experiments/IEEE-NS3
./ns3 run "scratch/leo-validation --input=$IN/packets_mappo.csv --output=$IN/dbg.csv --policy-name=mappo --bw-kbps=288 --qsize-pkts=64" 2>&1 | grep -E 'RESULT|scheduled|DBG|error|Error|assert|ASSERT'
