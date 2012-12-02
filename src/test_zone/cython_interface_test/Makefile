all:
	python Setup.py build_ext --inplace

test:	all
	python run_marchingcube.py

clean:
	@echo cleaning callback/interface demo
	@rm -f marchingcube_py.c *.o *.so *~ core
	@rm -rf build
