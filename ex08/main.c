#include <stdlib.h> // malloc, free, rand
#include <stdio.h>  // printf
#include <sys/time.h> // gettimeofday
#include <time.h>   // time
#include <pthread.h>

#define RED 0
#define BLACK 1



typedef enum { OP_ADD, OP_SEARCH } Operation;

typedef struct {
    Operation op;
    int value;
} UpdateAction;

typedef struct RBTNode {
    int data;
    int color; // RED = 0, BLACK = 1
    struct RBTNode *left, *right;
    pthread_rwlock_t lock;
} RBTNode;

typedef struct {
    RBTNode **root;
    UpdateAction *stream;
    int start;
    int end;
} ThreadArgs;

int search(RBTNode *root, int key) {
    RBTNode *curr = root;
    RBTNode *prev = NULL;

    if (!curr) return 0;

    pthread_rwlock_rdlock(&curr->lock);

    while (curr) {
        if (key < curr->data) {
            if (curr->left) pthread_rwlock_rdlock(&curr->left->lock);
            if (prev) pthread_rwlock_unlock(&prev->lock);
            prev = curr;
            curr = curr->left;
        } else if (key > curr->data) {
            if (curr->right) pthread_rwlock_rdlock(&curr->right->lock);
            if (prev) pthread_rwlock_unlock(&prev->lock);
            prev = curr;
            curr = curr->right;
        } else {
            pthread_rwlock_unlock(&curr->lock);
            if (prev) pthread_rwlock_unlock(&prev->lock);
            return 1;
        }
    }

    if (prev) pthread_rwlock_unlock(&prev->lock);
    return 0;
}


int isRed(RBTNode *node) {
    return node != NULL && node->color == RED;
}

RBTNode* rotateLeft(RBTNode *h) {
    RBTNode *x = h->right;
    h->right = x->left;
    x->left = h;
    x->color = h->color;
    h->color = RED;
    return x;
}

RBTNode* rotateRight(RBTNode *h) {
    RBTNode *x = h->left;
    h->left = x->right;
    x->right = h;
    x->color = h->color;
    h->color = RED;
    return x;
}

void flipColors(RBTNode *h) {
    h->color = RED;
    if (h->left) h->left->color = BLACK;
    if (h->right) h->right->color = BLACK;
}

RBTNode* createNode(int key) {
    RBTNode* node = malloc(sizeof(RBTNode));
    node->data = key;
    node->color = RED;
    node->left = node->right = NULL;
    pthread_rwlock_init(&node->lock, NULL);
    return node;
}

RBTNode* insertRecursiveLocked(RBTNode *h, int key, int *added) {
    pthread_rwlock_wrlock(&h->lock);

    if (key == h->data) {
        *added = 0;
        pthread_rwlock_unlock(&h->lock);
        return h;
    }

    if (key < h->data) {
        if (!h->left) {
            h->left = createNode(key);
            *added = 1;
        } else {
            pthread_rwlock_wrlock(&h->left->lock);
            pthread_rwlock_unlock(&h->lock);
            h->left = insertRecursiveLocked(h->left, key, added);
            // Do not re-lock parent or unlock child here
            return h; // Let recursion handle unlocking
        }
    } else {
        if (!h->right) {
            h->right = createNode(key);
            *added = 1;
        } else {
            pthread_rwlock_wrlock(&h->right->lock);
            pthread_rwlock_unlock(&h->lock);
            h->right = insertRecursiveLocked(h->right, key, added);
            return h;
        }
    }

    // Fix-up (rotations/flipColors) -- must ensure correct locking here!
    // For now, only do fix-up if we still hold the lock for h
    // (If you want to support concurrent rotations, you need to lock children as well)

    pthread_rwlock_unlock(&h->lock);
    return h;
}

void add(RBTNode **root, int key) {
    int added = 0;

    if (*root == NULL) {
        *root = createNode(key);
        (*root)->color = BLACK;
        return;
    }

    *root = insertRecursiveLocked(*root, key, &added);
    pthread_rwlock_wrlock(&(*root)->lock);
    (*root)->color = BLACK;
    pthread_rwlock_unlock(&(*root)->lock);
}

int checkRBTProperties(RBTNode *node, int blackCount, int *pathBlackCount) {
    if (!node) return 1;

    // Rule 1: Root is black
    // Rule 2: No red-red parent-child
    if (isRed(node)) {
        if (isRed(node->left) || isRed(node->right))
            return 0;
    }

    // Rule 3: All paths have same number of black nodes
    if (!isRed(node)) blackCount++;

    if (!node->left && !node->right) {
        if (*pathBlackCount == -1)
            *pathBlackCount = blackCount;
        else if (*pathBlackCount != blackCount)
            return 0;
    }

    return checkRBTProperties(node->left, blackCount, pathBlackCount) &&
           checkRBTProperties(node->right, blackCount, pathBlackCount);
}

int isValidRBT(RBTNode *root) {
    if (root == NULL) return 1;
    if (isRed(root)) return 0; // root must be black
    int pathBlackCount = -1;
    return checkRBTProperties(root, 0, &pathBlackCount);
}

// Lock-free, single-threaded RBT insert for initialization
RBTNode* insertRBTInit(RBTNode *h, int key, int *added) {
    if (!h) {
        *added = 1;
        RBTNode *n = createNode(key);
        return n;
    }
    if (key == h->data) {
        *added = 0;
        return h;
    }
    if (key < h->data)
        h->left = insertRBTInit(h->left, key, added);
    else
        h->right = insertRBTInit(h->right, key, added);

    // Fix-up: maintain RBT properties (left-leaning)
    if (isRed(h->right) && !isRed(h->left))
        h = rotateLeft(h);
    if (isRed(h->left) && isRed(h->left->left))
        h = rotateRight(h);
    if (isRed(h->left) && isRed(h->right))
        flipColors(h);

    return h;
}

void addRBTInit(RBTNode **root, int key) {
    int added = 0;
    *root = insertRBTInit(*root, key, &added);
    if (*root) (*root)->color = BLACK;
}

void init_rbt(RBTNode **root, int count) {
    srand(42); // Fixed seed for repeatability
    int inserted = 0;
    while (inserted < count) {
        int val = rand();  // Note: may need better uniqueness for 10M
        int wasFound = search(*root, val);
        if (!wasFound) {
            addRBTInit(root, val);
            inserted++;
        }
    }
}

void free_rbt(RBTNode *node) {
    if (!node) return;
    free_rbt(node->left);
    free_rbt(node->right);
    pthread_rwlock_destroy(&node->lock); // Destroy lock
    free(node);
}

UpdateAction* generate_update_stream(int stream_size, float insert_ratio) {
    UpdateAction *stream = malloc(sizeof(UpdateAction) * stream_size);
    srand(time(NULL));
    for (int i = 0; i < stream_size; i++) {
        float r = (float)rand() / RAND_MAX;
        stream[i].op = (r < insert_ratio) ? OP_ADD : OP_SEARCH;
        stream[i].value = rand(); // You can skew this if needed
    }
    return stream;
}

double current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

void *thread_worker(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    for (int i = args->start; i < args->end; i++) {
        if (args->stream[i].op == OP_ADD)
            add(args->root, args->stream[i].value);
        else
            search(*(args->root), args->stream[i].value);
    }
    return NULL;
}

void threaded_benchmark(RBTNode **root, int stream_size, float insert_ratio, int num_threads) {
    UpdateAction *stream = generate_update_stream(stream_size, insert_ratio);
    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];

    int chunk_size = stream_size / num_threads;

    double start = current_time();

    for (int i = 0; i < num_threads; i++) {
        args[i].root = root;
        args[i].stream = stream;
        args[i].start = i * chunk_size;
        args[i].end = (i == num_threads - 1) ? stream_size : (i + 1) * chunk_size;
        pthread_create(&threads[i], NULL, thread_worker, &args[i]);
    }

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    double end = current_time();

    double duration = end - start;
    printf("%d,%d,%.4f,%.2f\n", stream_size, num_threads, duration, stream_size / duration);

    free(stream);
}


int main(int argc, char *argv[]) {
    int thread_counts[] = {2, 4, 8, 12, 16, 24, 32, 48};
    int stream_sizes[] = {100000, 500000, 1000000, 5000000, 10000000, 25000000, 50000000};
    int num_sizes = sizeof(stream_sizes) / sizeof(stream_sizes[0]);
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    int initial_elements = 10000000;
    float insert_ratio = 0.10; // 10% add, 90% search

    if (argc >= 2)
        insert_ratio = atof(argv[1]); // allow ratio override

    RBTNode *root = NULL;

    printf("stream_size,num_threads,duration,ops_per_sec\n");

    for (int i = 0; i < num_sizes; i++) {
        for (int j = 0; j < num_threads; j++) {
            RBTNode *root = NULL;
            init_rbt(&root, 10000000); // always start from 10M base
            if (!isValidRBT(root)) {
                printf("Invalid RBT!\n");
                return 1;
            }
            threaded_benchmark(&root, stream_sizes[i], insert_ratio, thread_counts[j]);
            free_rbt(root);
        }
    }

    return 0;
}
