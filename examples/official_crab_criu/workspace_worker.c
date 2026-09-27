/* Owned RAM + actual held-open rootfs file, quiescent at sigwaitinfo.
 * This fixture has no threads, network, inherited host-connected stdio, or
 * child exec in flight. It is not a graded software repository task.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void accepted_signal(int number) { (void)number; }

static int write_all(int fd, const void *source, size_t length) {
    const unsigned char *data = source;
    while (length) {
        ssize_t count = write(fd, data, length);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return -1;
        data += count;
        length -= (size_t)count;
    }
    return 0;
}

int main(int argc, char **argv) {
    const long mib = argc == 2 ? strtol(argv[1], NULL, 10) : 8;
    if (mib < 1 || mib > 64) return 19;
    int sink = open("/dev/null", O_RDWR);
    if (sink < 0) return 20;
    for (int fd = 0; fd < 3; fd++) if (dup2(sink, fd) < 0) return 20;
    if (sink > 2 && close(sink) != 0) return 20;
    sigset_t signals;
    struct sigaction disposition;
    memset(&disposition, 0, sizeof(disposition));
    disposition.sa_handler = accepted_signal;
    sigemptyset(&disposition.sa_mask);
    if (sigaction(SIGUSR1, &disposition, NULL) != 0 ||
        sigaction(SIGUSR2, &disposition, NULL) != 0) return 21;
    sigemptyset(&signals);
    sigaddset(&signals, SIGUSR1);
    sigaddset(&signals, SIGUSR2);
    if (sigprocmask(SIG_BLOCK, &signals, NULL) != 0) return 21;
    const size_t size = (size_t)mib * 1024 * 1024;
    const long page_size = sysconf(_SC_PAGESIZE);
    unsigned char *pages = mmap(NULL, size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED || page_size <= 0) return 22;
    for (size_t offset = 0; offset < size; offset += (size_t)page_size)
        pages[offset] = 23;
    volatile uint64_t *counter = (uint64_t *)pages;
    volatile uint64_t *last = (uint64_t *)(pages + size - (size_t)page_size);
    *counter = 41;
    *last = 11;
    const int ledger = open("/workspace/ledger.bin", O_CREAT | O_EXCL | O_RDWR, 0600);
    if (ledger < 0) return 23;
    unsigned char contents[512];
    for (size_t i = 0; i < sizeof(contents); i++) contents[i] = (unsigned char)((13 * i + 7) % 251);
    if (write_all(ledger, contents, sizeof(contents)) != 0 || fsync(ledger) != 0 ||
        lseek(ledger, 64, SEEK_SET) != 64) return 24;
    int saved = open("/workspace/saved.txt", O_CREAT | O_EXCL | O_WRONLY, 0600);
    static const char saved_bytes[] = "saved-at-phase-41\n";
    if (saved < 0 || write_all(saved, saved_bytes, sizeof(saved_bytes) - 1) != 0 ||
        fsync(saved) != 0 || close(saved) != 0) return 25;
    struct stat object;
    if (fstat(ledger, &object) != 0) return 26;
    char identity[512];
    int length = snprintf(identity, sizeof(identity),
        "{\"address\":%llu,\"bytes\":%zu,\"page_size\":%ld,"
        "\"held_fd\":%d,\"file_inode\":%llu,\"file_device\":%llu,"
        "\"namespace_pid\":%ld,\"initial_file_offset\":64}\n",
        (unsigned long long)(uintptr_t)pages, size, page_size, ledger,
        (unsigned long long)object.st_ino, (unsigned long long)object.st_dev,
        (long)getpid());
    int identity_fd = open("/probe/identity.json", O_CREAT | O_EXCL | O_WRONLY, 0600);
    if (identity_fd < 0 || length < 0 || (size_t)length >= sizeof(identity) ||
        write_all(identity_fd, identity, (size_t)length) != 0 ||
        fsync(identity_fd) != 0 || close(identity_fd) != 0) return 27;
    for (;;) {
        int number = sigwaitinfo(&signals, NULL);
        if (number == SIGUSR1) {
            (*counter)++;
            char record[16];
            int count = snprintf(record, sizeof(record), "PH%06llu", (unsigned long long)*counter);
            if (count != 8 || write_all(ledger, record, 8) != 0 || fsync(ledger) != 0) return 28;
        } else if (number == SIGUSR2) {
            (*last)++;
        } else if (number < 0 && errno != EINTR) {
            return 29;
        }
    }
}
