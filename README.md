# nvjpeg2k_numcodecs

This is a playground for getting nvjpeg2k decoding into numcodecs to
benchmark if tile loading for deeplearning can be significantly sped
up when decoding directly on the GPU.

## :construction: Notes :construction:

Currently, this is a bit of a playground for testing feasibility and
establishing good benchmarks. But feel free to open issues in the
issue tracker in case you would like to help and or contribute :heart:

## Documentation

### Installation

Currently no wheels or conda packages.
To get started you need cuda and a cupy installation.
Then clone the repo and run:

```shell
cd vendor
./get_nvjpeg2k.sh
```

after that you should be able to run the following in the repository root:

```shell
pip install .
```


## Contributing Guidelines

Please use `pre-commit` unless it interferes with your workflow too much.
I'm happy to get contributions, so don't let linting be in the way of that.

## Acknowledgements

Build with love by Andreas Poehlmann.

`nvjpeg2k_numcodecs`: copyright 2022 Andreas Poehlmann, licensed under [MIT](https://github.com/ap--/nvjpeg2k_numcodecs/blob/master/LICENSE)
