#!/bin/bash

# Docker build command
docker build -t digiclf:v3 -f docker/Dockerfile .

# Docker run command
docker run -v /home/sauravrvce/mlops/digit-classification/models:/models digiclf:v3
