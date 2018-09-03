# Add a new filepath to pythonpath only if it is new
pythonpathmunge () {
        if ! echo "$PYTHONPATH" | /bin/grep -Eq "(^|:)$1($|:)" ; then
           if [ "$2" = "after" ] ; then
              PYTHONPATH="$PYTHONPATH:$1"
           else
              PYTHONPATH="$1:$PYTHONPATH"
           fi
        fi
}
pythonpathmunge $PWD

export PYTHONPATH=$PYTHONPATH
