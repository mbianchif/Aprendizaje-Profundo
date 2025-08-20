def main():
    images = retrieve_images("images")
    print(images.__iter__().__next__())


if __name__ == "__main__":
    main()
