for file in *.tar ; do
    # strip any leading path information from the file name
    name_only="${file##*/}"

    # strip the extension .tar.gz
    without_extension="${name_only%.tar}"

    # make a directory named after the tar file, minus extension
    mkdir -p "$without_extension"

    # extract the file into its own directory
    tar -xvf "$file" -C "$without_extension"

    rm "$file"
done
