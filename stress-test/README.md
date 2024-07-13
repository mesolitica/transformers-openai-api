# Stress test

## T5 on RTX 3090 Ti

Rate of 5 users per second, total requests up to 50 users for 60 seconds,

```bash
locust -f t5_continous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
locust -f t5_without_continous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
```

### Non-continous batch

![alt text](t5_without_continous.png)

### Continous batch

![alt text](t5_continous.png)

## Mistral 7b GPTQ on RTX 3090 Ti

Rate of 5 users per second, total requests up to 50 users for 60 seconds,

```bash
locust -f mistral_7b_gtpq_continous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
locust -f mistral_7b_gtpq_without_continous.py -P 7001 -H http://localhost:7088 -r 5 -u 50 -t 60
```

### Non-continous batch

![alt text](mistral_7b_gtpq_without_continous.png)

### Continous batch

![alt text](mistral_7b_gtpq_continous.png)