## To build the documentation

Install sphinx:
```shell script
apt-get install sphinx
```

From the `docs/` directory:
```shell script
sphinx-apidoc -f -o source/_build/ ../
make html
```

Open the documentation using:
```shell script
firefox build/html/index.html
```

TODO: Integrate with Github pages