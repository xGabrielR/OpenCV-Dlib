## Opencv Dlib

---

### Project Description

![image](https://github.com/xGabrielR/OpenCV-Dlib/assets/75986085/5fc37c24-bdda-4f7e-b314-9ca6784ddfb4)


This project has a very simple idea. It basically consists of a camera detecting people's faces using yunet, after that the image is processed so that the embedding is collected with Dlib, and this 128 position embedding vector is stored in a document in Elasticsearch for quick query/similarity of images later.
The idea of ​​using c++ was to test these tools instead of doing it in python.

Elasticsearch is an open-source, highly scalable search and analytics engine built on top of the Apache Lucene library. It is designed to handle large volumes of structured and unstructured data and provide fast and efficient searching, indexing, and data analysis capabilities.

At its core, Elasticsearch stores data in a distributed and schema-free manner, allowing for real-time search and analysis across a wide range of data types. It uses a document-oriented approach, where data is stored as JSON documents that can be easily indexed and queried.

One of the key features of Elasticsearch is its distributed nature. It employs a distributed architecture that allows data to be divided and replicated across multiple nodes in a cluster. This enables Elasticsearch to handle high volumes of data and provide high availability and fault tolerance. It also allows for horizontal scalability, where additional nodes can be added to the cluster to increase storage capacity and processing power.

### Requirements

---

1. Dlib for dlib encoding net.
2. Opencv for camera recording.
3. Yunet face detector.
4. Elasticlient instance for store image embeddings.


### Elasticsearch Simple Guide

---

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

**Insert Data on Index**.

```
curl -H "Content-Type: application/json" -XPOST 127.0.0.1:9200/img_emb/_doc -d '
    {"vector": [0.254, 0.123, ..., -0.94]}'
```

### References

---

[1] [elasticlient](https://github.com/seznam/elasticlient): Elasticlient C++ Library.

[2] [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet): YuNet opencv_zoo Repo.
