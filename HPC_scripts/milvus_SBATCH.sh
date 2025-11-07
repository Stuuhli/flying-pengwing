#!/bin/bash
export ETCD_USE_EMBED=true
export ETCD_DATA_DIR=/var/lib/milvus/etcd
export ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml
export COMMON_STORAGETYPE=local
cd /milvus && ./bin/milvus run standalone