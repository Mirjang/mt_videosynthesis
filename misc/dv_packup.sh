set -ex
rm -rf dl
mkdir dl
cp $1/*.png dl/
zip -r results.zip dl
rm -r dl
