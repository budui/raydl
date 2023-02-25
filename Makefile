bumpver:
	bumpver update  --patch -n

upload:
	rm build/*
	python -m build
	twine upload  dist/*
