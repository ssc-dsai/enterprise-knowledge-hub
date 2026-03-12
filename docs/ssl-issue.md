# SSL Issues

Here is how to fix the corporate SSL issues. This fix would apply to any ICM/or 
other certificates that need to be included in the certificate chain. This assumes you are using linux to work.

## How to get certs

First get the certificate that is intercepting your traffic (e.g. corporate proxy/firewall certificates).

This is done via `openssl` command: 

```bash
openssl s_client -connect huggingface.io:443 -showcerts
```

Then you can grab certs you need from there and follow instructions bellow.

## How to fix

The certificates are stored in the `./certs` directory:
- `1ca-ac1-2026.pem`
- `1ca-ac1-2037.pem`

## Install the certificates (Ubuntu)

Combine both PEM files into a single `.crt` file and copy it to the trusted certificates directory:

```bash
cat ./certs/1ca-ac1-2026.pem ./certs/1ca-ac1-2037.pem > /usr/local/share/ca-certificates/corporate-ca-bundle.crt
```

Then update the system CA store:

```bash
sudo update-ca-certificates
```