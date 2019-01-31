#!/bin/bash

# Do not uncomment these lines to directly execute the script
# Modify the path to fit your need before using this script

SIZE=$1
ITER=$2
JAR=PageRank.jar

hadoop jar $JAR pagerank.PageRank $SIZE $ITER
hdfs dfs -getmerge pr_$1/result ~/homework/HW5/pagerank_$1.out
