# Fetal Brain Mask

[![Version](https://img.shields.io/docker/v/fnndsc/pl-fetal-brain-mask?sort=semver)](https://hub.docker.com/r/fnndsc/pl-fetal-brain-mask)
[![MIT License](https://img.shields.io/github/license/fnndsc/pl-fetal-brain-mask)](https://github.com/FNNDSC/pl-fetal-brain-mask/blob/master/LICENSE)

Automatic brain image masking tool for fetal brain MRI using deep learning.

For each `.nii` or `.nii.gz` file in a directory, a `_mask.nii` is produced.

![Figure](docs/fetal_brain_mask.png)

##  Usage

```
$ docker run --rm fnndsc/pl-fetal-brain-mask:1.0.0 fetal_brain_mask --help 2> /dev/null

usage: fetal_brain_mask [-h] [--json] [--savejson DIR] [--inputmeta INPUTMETA]
                        [--saveinputmeta] [--saveoutputmeta] [--version]
                        [--meta] [-v VERBOSITY] [--man] [-p INPUTPATHFILTER]
                        [-o SUFFIX] [-s] [-l SKIPPED_LIST]
                        inputdir outputdir

Automatic masking of fetal brain images using deep learning

positional arguments:
  inputdir              directory containing the input files
  outputdir             directory containing the output files/folders

optional arguments:
  -h, --help            show this help message and exit
  --json                show json representation of app and exit
  --version             print app version and exit
  --meta                print app meta data and exit
  -v VERBOSITY, --verbosity VERBOSITY
                        verbosity level for the app (default: 0)
  --man                 show the app's man page and exit (default: False)
  -p INPUTPATHFILTER, --inputPathFilter INPUTPATHFILTER
                        selection for which files to evaluate (default:
                        **.nii[,.gz])
  -o SUFFIX, --suffix SUFFIX
                        output filename suffix (default: _mask.nii)
  -s, --smooth          perform post-processing on images including
                        morphological closing and defragmentation (default:
                        True)
  -l SKIPPED_LIST, --skipped-list SKIPPED_LIST
                        produce an output file containing the names of files
                        which were skipped (default: )
```

## Examples

Single sample, minimum example

```bash
mkdir in out
mv scan.nii.gz in

docker run --rm  \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw  \
    fnndsc/pl-fetal-brain-mask:1.0.0 fetal_brain_mask \
    /incoming /outgoing
```

User-friendly settings for Docker

```bash
docker run --rm -u $(id -u):$(id -g)                  \
    -v /etc/localtime:/etc/localtime:ro               \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw  \
    fnndsc/pl-fetal-brain-mask:1.0.0 fetal_brain_mask \
    /incoming /outgoing
```

Specifying filenames

```bash
docker run --rm -u $(id -u):$(id -g)                  \
    -v /etc/localtime:/etc/localtime:ro               \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw  \
    fnndsc/pl-fetal-brain-mask:1.0.0 fetal_brain_mask \
    --inputPathFilter scan.nii.gz --suffix _mask.nii  \
    --skipped-list skipped.txt                        \
    /incoming /outgoing

```

Multiple inputs are processed in parallel, using all the cores visible inside the container.
For large datasets, you can limit the number of concurrent jobs to `N` jobs with `docker run --cpuset-cpus 0-$((N-1))`

```bash
mkdir in out
mv scan0001.nii.gz scan0002.nii.gz scan0003.nii.gz scan0004.nii.gz \
   scan0005.nii.gz scan0006.nii.gz scan0007.nii.gz scan0008.nii.gz in

# limit to 5 concurrent jobs
docker run --rm -u $(id -u):$(id -g)                   \
    -v /etc/localtime:/etc/localtime:ro                \
    --cpuset-cpus 0-4                                  \
    -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw   \
    fnndsc/pl-fetal-brain-mask:1.0.0 fetal_brain_mask  \
    --inputPathFilter 'scan????.nii.gz'                \
    --skipped-list skipped.txt                         \
    /incoming /outgoing
```

See `docker run --help` for other throttling options.

Some (not all) unnessessary tensorflow output can be silcence

```bash
docker run --rm -e TF_CPP_MIN_LOG_LEVEL=3 fnndsc/pl-fetal-brain-mask:1.0.0
```

## Links

Original source:
https://github.com/chrisorozco1097/masking_tool
