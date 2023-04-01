bumpver:
	bumpver update  --patch -n

upload:
	rm -r build
	python -m build
	twine upload  dist/*
