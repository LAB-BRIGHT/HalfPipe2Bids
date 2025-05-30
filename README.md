# HalfPipe2BIDS

HalfPipe2Bids is a tool designed to convert neuroimaging data from the **HalfPipe** format to the standardized **BIDS** (Brain Imaging Data Structure) format.

<!-- 
## Table of Contents <!-- omit in toc -->
- [Features](#features)
- [Requirements](#requirements)
- [Running HalfPipe2BIDS](#running-halfpipe2bids)
  - [Using Docker](#using-docker)
  - [Using Singularity](#using-singularity)
- [Contributing](#contributing)
- [License](#license)


## Features
- Converts HalfPipe data to BIDS format
- Multi-platform support via Docker and Singularity
- Detailed logs to track the conversion process


## Requirements
Before running HalfPipe2BIDS, ensure the following dependencies are installed:
- [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/guides/3.8/user-guide/installation.html)
- [Make](https://www.gnu.org/software/make/) 
-->

## Running HalfPipe2BIDS

Install python package:

```
git clone git@github.com:pbergeret12/HalfPipe2Bids.git
pip install -e .
```

Running the CLI:
```
halfpipe2bids halfpipe2bids/tests/data/dataset-ds000030_halfpipe1.2.3dev outputs group
```
<!-- 
### Using Docker
### Using Singularity

## Contributing

## License -->
