set -ex
rm -rf dl
mkdir dl
cp $1/images/*_fake* dl
zip -r results.zip dl
rm -r dl
