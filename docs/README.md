## To build the documentation

Install sphinx:
```shell script
apt-get install sphinx
```

From the `docs/` directory:
```shell script
cd source
sphinx-apidoc -f -o . ../..
cd ../
make html
```

Open the documentation using:
```shell script
firefox build/html/index.html
```

TODO: Integrate with Github pages