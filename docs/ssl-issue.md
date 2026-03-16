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

First add this to the `.env` file (see `.env.example` for more examples):

```bash
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```
Note: You can also `vi ~/.bashrc` (or `.zshrc`) and `export BLAH=var` the variables. Don't forget to `source ~/.bashrc`
after for the changes to take effect in your current terminal.

## Install the certificates (Ubuntu)

Copy `GOC-GDC-ROOT-A.crt` in `GOC-GDC-ROOT-A.crt and update the trust bundle.

```bash
cp ./certs/GOC-GDC-ROOT-A.crt /usr/local/share/ca-certificates/GOC-GDC-ROOT-A.crt
sudo update-ca-certificates
```