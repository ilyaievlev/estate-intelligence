NAMESPACE ?= estate
RELEASE   ?= estate

.PHONY: up build forward down destroy status logs test traffic

up:            ## поднять всё и держать port-forward
	./scripts/up.sh

build:         ## пересобрать образы и передеплоить
	./scripts/up.sh --build

forward:       ## только port-forward
	./scripts/port_forward.sh

status:
	kubectl get pods,hpa,svc,pvc -n $(NAMESPACE)

logs:          ## make logs app=inference
	kubectl logs -n $(NAMESPACE) -l app=$(app) -f --tail=200

down:          ## удалить релиз, данные в PVC остаются
	helm uninstall $(RELEASE) -n $(NAMESPACE)

destroy:       ## удалить всё вместе с данными
	kubectl delete namespace $(NAMESPACE)

test:
	services/collector/.venv/bin/python services/inference/test_api.py

traffic:
	services/collector/.venv/bin/python scripts/mock_traffic.py
