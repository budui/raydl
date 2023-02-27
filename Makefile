bumpver:
	bumpver update  --patch -n

upload:
	python -m build
	twine upload  dist/*
