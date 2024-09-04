# export PROJECT_ROOT=~/MARBLE-Benchmark

#cd $PROJECT_ROOT
mkdir -p data/musdb18
cd data/musdb18

wget https://zenodo.org/record/1117372/files/musdb18.zip
unzip musdb18.zip
rm musdb18.zip
mv musdb18 musdb18_mp4
cd ..

#musdbconvert musdb18 musdb18_wav
#rm -r musdb18
#mv musdb18_wav musdb18
#
#cd $PROJECT_ROOT