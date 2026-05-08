#!/bin/bash
# run spain-container in interactive mode and mount this directory to home/reviewer/app
docker run -it --rm \
  -v "$(pwd)":/home/reviewer/app \
  -v cargo-registry:/usr/local/cargo/registry \
  -v cargo-git:/usr/local/cargo/git \
  -w /home/reviewer/app \
  --platform linux/amd64 \
  spain-container
