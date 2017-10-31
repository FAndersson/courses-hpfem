# Howto start a Jupyter notebook server on a local host

```
docker run --rm -d -ti -p 127.0.0.1:8888:8888 --name notebook quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
```

# Get notebook token

```
docker exec -it -u fenics notebook /bin/bash
jupyter notebook list
```
