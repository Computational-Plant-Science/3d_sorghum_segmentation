# run pipeline
python3 /opt/code/ai_seg.py -i $INPUT -o $OUTPUT 

# copy nested output files to working directory
#find . -type f -name "*_masked.jpg" -exec cp {} $WORKDIR \;


