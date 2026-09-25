deploy:
	scripts/deploy.sh all
worker:
	scripts/deploy.sh worker
image:
	scripts/deploy.sh image
status:
	scripts/ops.sh status
run-now:
	scripts/ops.sh run-now
e2e:
	scripts/e2e.sh
test:
	cd pod && python -m pytest ../tests -q && cd ../worker && npx tsc --noEmit
