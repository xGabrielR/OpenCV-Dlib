## Opencv Dlib

---

### Project Description

This project has a very simple idea. It basically consists of a camera detecting people's faces using yunet, after that the image is processed so that the embedding is collected with Dlib, and this 128 position embedding vector is stored in a document in Elasticsearch for quick query/similarity of images later.
The idea of ​​using c++ was to test these tools instead of doing it in python.

Requirements:

1. Dlib for dlib encoding net.
2. Opencv for camera recording.
3. Elasticlient for store image encoding.

Elastic Vector:

**Create Index**

```
curl -H "Content-Type: application/json" -XPUT 127.0.0.1:9200/img_emb -d '{
    "mappings": {
        "properties": {
            "vector": {
                "dims": 128,
                "type": "dense_vector",
                "similarity": "l2_norm"
            }
        }
    }
}'
```

**Check Index Properties**

```
curl -H "Content-Type: application/json" -XGET 127.0.0.1:9200/img_emb
```

**Insert Data on Index**

Generated Vector of Process.

```
curl -H "Content-Type: application/json" -XPOST 127.0.0.1:9200/img_emb/_doc -d '
    {"vector": [0.254, 0.123, ..., -0.94]}'
```

References:

https://github.com/seznam/elasticlient
