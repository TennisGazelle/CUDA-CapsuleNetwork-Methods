#!/bin/bash
tail -c+1 -f ga_best_config.out | grep --line-buffered Loss | awk '{ print $3 }'
