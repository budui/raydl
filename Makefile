bumpver:
	bumpver update  --patch -n

upload:
	rm -rf build
	python -m build
	twine upload  dist/*
