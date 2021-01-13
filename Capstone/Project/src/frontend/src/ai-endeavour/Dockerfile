FROM nginx:alpine
LABEL author="Praveen Raghuvanshi"
COPY ./dist/browser /usr/share/nginx/html
EXPOSE 80 443
ENTRYPOINT [ "nginx", "-g", "daemon off;" ]
